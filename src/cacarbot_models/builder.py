"""Builds and compiles models to .vmfb format for the Coralboard SL NPU."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional

from .models import ModelInfo


def check_tool(name: str) -> Optional[str]:
    """Check if an executable is available in the current environment."""
    return shutil.which(name)


def check_torq_compiler() -> Optional[str]:
    """Check if torq-compile is available in the current environment."""
    return check_tool("torq-compile")


def patch_yolox_focus_layer(input_onnx: Path, output_onnx: Path) -> bool:
    """Replace the slicing Focus layer with a static Reshape-Transpose-Reshape sequence."""
    try:
        import numpy as np
        import onnx
        from onnx import helper, TensorProto

        model = onnx.load(str(input_onnx))
        if len(model.graph.node) < 9:
            return False

        # Check if first 8 nodes are Slice and 9th is Concat
        first_ops = [n.op_type for n in model.graph.node[:9]]
        if first_ops != ["Slice"] * 8 + ["Concat"]:
            # Model does not have the standard YOLOX slicing Focus layer
            return False

        input_tensor = model.graph.input[0]
        input_dims = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
        if len(input_dims) != 4 or not input_dims[2] or not input_dims[3]:
            return False

        b, c, h, w = input_dims
        h_half, w_half = h // 2, w // 2
        concat_out = model.graph.node[8].output[0]

        shape1_init = helper.make_tensor("focus_shape1", TensorProto.INT64, [6], [b, c, h_half, 2, w_half, 2])
        shape2_init = helper.make_tensor("focus_shape2", TensorProto.INT64, [4], [b, c * 4, h_half, w_half])
        model.graph.initializer.extend([shape1_init, shape2_init])

        node_r1 = helper.make_node("Reshape", [input_tensor.name, "focus_shape1"], ["focus_r1"], name="CleanFocus_R1")
        node_trans = helper.make_node(
            "Transpose", ["focus_r1"], ["focus_trans"], perm=[0, 5, 3, 1, 2, 4], name="CleanFocus_Trans"
        )
        node_r2 = helper.make_node("Reshape", ["focus_trans", "focus_shape2"], [concat_out], name="CleanFocus_R2")

        new_nodes = [node_r1, node_trans, node_r2] + list(model.graph.node[9:])
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

        onnx.checker.check_model(model)
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(output_onnx))
        return True
    except Exception as e:
        print(f"[NOTE] Could not patch Focus layer in ONNX: {e}")
        return False


def build_model(model_info: ModelInfo, input_path: Path, output_dir: Path) -> Optional[Path]:
    """Compile an ONNX, TFLite, or MLIR model into a Coralboard SL .vmfb binary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_vmfb = output_dir / f"{model_info.name}.vmfb"

    # Pre-compiled VMFBs copy directly
    if model_info.is_precompiled_vmfb or input_path.suffix == ".vmfb":
        if input_path != target_vmfb:
            shutil.copyfile(input_path, target_vmfb)
        print(f"[OK] Model {model_info.name} is pre-compiled and ready: {target_vmfb}")
        return target_vmfb

    # Check if target .vmfb already exists
    if target_vmfb.exists() and target_vmfb.stat().st_size > 0:
        print(f"[OK] Already compiled: {target_vmfb.name} ({target_vmfb.stat().st_size / (1024*1024):.1f} MB)")
        return target_vmfb

    compiler = check_tool("torq-compile")
    if not compiler:
        print("\n" + "=" * 70)
        print(f"[INFO] Downloaded {model_info.display_name} successfully to:")
        print(f"   {input_path}")
        print("\n[NOTE] 'torq-compile' was not found on your system PATH.")
        print("   To compile this model into a '.vmfb' binary for the Coralboard SL NPU:")
        print("   1. Run with the optional compile dependency on an x86_64 system:")
        print("      uv run --extra compile cacarbot-models")
        print("   2. Or install compilation tools manually:")
        print("      pip install torq-compiler tosa-converter-for-tflite onnx2tf[tensorflow]")
        print("=" * 70 + "\n")
        return None

    work_dir = output_dir / f".work_{model_info.name}"
    work_dir.mkdir(parents=True, exist_ok=True)

    tosa_mlir: Optional[Path] = None

    # Step 1: Handle ONNX inputs
    if input_path.suffix.lower() == ".onnx":
        clean_onnx = work_dir / f"{model_info.name}_clean.onnx"
        if not clean_onnx.exists():
            print(f"Optimizing ONNX graph for {model_info.name}...")
            if not patch_yolox_focus_layer(input_path, clean_onnx):
                shutil.copyfile(input_path, clean_onnx)

        onnx2tf_tool = check_tool("onnx2tf")
        if not onnx2tf_tool:
            print("[ERROR] 'onnx2tf' is required to quantize ONNX models for Torq NPU.")
            print("   Install via: uv run --extra compile cacarbot-models")
            return None

        # Inspect model input tensor name and spatial dimensions
        input_name = "images"
        calib_h, calib_w = 416, 416
        try:
            import onnx

            model_proto = onnx.load(str(clean_onnx))
            if model_proto.graph.input:
                first_input = model_proto.graph.input[0]
                input_name = first_input.name
                dims = [d.dim_value for d in first_input.type.tensor_type.shape.dim]
                if len(dims) == 4 and dims[2] > 0 and dims[3] > 0:
                    calib_h, calib_w = dims[2], dims[3]
        except Exception:
            pass

        # Generate sample calibration dataset
        calib_path = work_dir / f"calibration_{calib_h}x{calib_w}.npy"
        if not calib_path.exists():
            try:
                import numpy as np

                calib_data = np.random.uniform(0.0, 1.0, size=(10, calib_h, calib_w, 3)).astype(np.float32)
                np.save(str(calib_path), calib_data)
            except Exception as e:
                print(f"[ERROR] Failed to generate calibration data: {e}")
                return None

        tflite_dir = work_dir / "tflite_int8"
        tflite_file = tflite_dir / f"{clean_onnx.stem}_full_integer_quant.tflite"

        if not tflite_file.exists():
            print(f"Quantizing {model_info.name} to INT8 using onnx2tf...")
            cmd_onnx2tf = [
                onnx2tf_tool,
                "-i",
                str(clean_onnx),
                "-o",
                str(tflite_dir),
                "-oiqt",
                "-iqd",
                "int8",
                "-oqd",
                "int8",
                "--tflite_backend",
                "tf_converter",
                "-cind",
                input_name,
                str(calib_path),
                "[[[[0.0,0.0,0.0]]]]",
                "[[[[1.0,1.0,1.0]]]]",
                "--non_verbose",
            ]
            res_tf = subprocess.run(cmd_onnx2tf, capture_output=True, text=True, check=False)
            if res_tf.returncode != 0 or not tflite_file.exists():
                print(f"[ERROR] onnx2tf quantization failed for {model_info.name}:")
                if res_tf.stderr:
                    print(res_tf.stderr[-1000:])
                return None

        tosa_converter = check_tool("tosa-converter-for-tflite")
        if not tosa_converter:
            print("[ERROR] 'tosa-converter-for-tflite' is required to compile TFLite models.")
            return None

        tosa_mlir = work_dir / f"{model_info.name}.tosa.mlir"
        if not tosa_mlir.exists():
            print(f"Converting INT8 TFLite to TOSA MLIR for {model_info.name}...")
            res_tosa = subprocess.run(
                [tosa_converter, str(tflite_file), "--text", "-o", str(tosa_mlir)],
                capture_output=True,
                text=True,
                check=False,
            )
            if res_tosa.returncode != 0 or not tosa_mlir.exists():
                print(f"[ERROR] tosa-converter-for-tflite failed for {model_info.name}:")
                if res_tosa.stderr:
                    print(res_tosa.stderr)
                return None

    # Step 2: Handle TFLite inputs directly
    elif input_path.suffix.lower() == ".tflite":
        tosa_converter = check_tool("tosa-converter-for-tflite")
        if not tosa_converter:
            print("[ERROR] 'tosa-converter-for-tflite' is required to compile TFLite models.")
            return None

        tosa_mlir = work_dir / f"{model_info.name}.tosa.mlir"
        if not tosa_mlir.exists():
            print(f"Converting TFLite to TOSA MLIR for {model_info.name}...")
            res_tosa = subprocess.run(
                [tosa_converter, str(input_path), "--text", "-o", str(tosa_mlir)],
                capture_output=True,
                text=True,
                check=False,
            )
            if res_tosa.returncode != 0 or not tosa_mlir.exists():
                print(f"[ERROR] tosa-converter-for-tflite failed for {model_info.name}:")
                if res_tosa.stderr:
                    print(res_tosa.stderr)
                return None

    elif input_path.suffix.lower() in [".tosa", ".mlir"]:
        tosa_mlir = input_path

    if not tosa_mlir or not tosa_mlir.exists():
        print(f"[ERROR] No valid TOSA MLIR intermediate file available for {model_info.name}.")
        return None

    # Step 3: Run torq-compile
    print(f"Compiling {model_info.name} with torq-compile for SL2610 NPU...")
    cmd_compile = [
        compiler,
        str(tosa_mlir),
        "-o",
        str(target_vmfb),
        "--torq-hw=SL2610",
    ]

    try:
        result = subprocess.run(cmd_compile, capture_output=True, text=True, check=False)
        if result.returncode == 0 and target_vmfb.exists() and target_vmfb.stat().st_size > 0:
            print(f"[OK] Successfully compiled {model_info.name} -> {target_vmfb}")
            return target_vmfb
        else:
            print(f"[ERROR] torq-compile failed for {model_info.name}:")
            if result.stderr:
                print(result.stderr[-1000:])
            return None
    except Exception as e:
        print(f"[ERROR] Error invoking torq-compile: {e}")
        return None
