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


def prepare_yolov6_heads(input_onnx: Path, output_onnx: Path) -> bool:
    """Decouple YOLOv6 output into separate boxes and scores heads for independent INT8 quantization."""
    try:
        import onnx
        from onnx import helper, TensorProto

        model = onnx.load(str(input_onnx))
        if not model.graph.node:
            return False

        last_node = model.graph.node[-1]
        # In YOLOv6 export, last node is Concat combining boxes, objectness, and class scores
        if last_node.op_type == "Concat" and len(last_node.input) >= 3:
            box_tensor = last_node.input[0]
            scores_tensor = last_node.input[2]
            model.graph.node.remove(last_node)

            del model.graph.output[:]
            out_b = helper.make_tensor_value_info(box_tensor, TensorProto.FLOAT, [None, 8400, 4])
            out_s = helper.make_tensor_value_info(scores_tensor, TensorProto.FLOAT, [None, 8400, 80])
            model.graph.output.extend([out_b, out_s])

            onnx.checker.check_model(model)
            output_onnx.parent.mkdir(parents=True, exist_ok=True)
            onnx.save(model, str(output_onnx))
            return True
        return False
    except Exception as e:
        print(f"[NOTE] Could not decouple YOLOv6 heads in ONNX: {e}")
        return False


def prune_ppyoloe_postprocessing(input_onnx: Path, output_onnx: Path) -> bool:
    """Prune dynamic NMS and post-processing layers from PP-YOLOE+ and expose static box/score heads."""
    try:
        import onnx
        from onnx import helper, TensorProto

        model = onnx.load(str(input_onnx))
        nms_idx = None
        for i, n in enumerate(model.graph.node):
            if n.op_type == "NonMaxSuppression":
                nms_idx = i
                break
        if nms_idx is None:
            return False

        # Identify box generator and score generator inputs to NMS
        nms_node = model.graph.node[nms_idx]
        box_in = nms_node.input[0]
        score_in = nms_node.input[1]

        # Prune everything starting from NMS
        del model.graph.node[nms_idx:]

        del model.graph.output[:]
        out_boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [None, 8400, 4])
        out_scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, 80, 8400])

        for n in model.graph.node:
            for idx, out in enumerate(n.output):
                if out == box_in:
                    n.output[idx] = "boxes"
                elif out == score_in:
                    n.output[idx] = "scores"

        model.graph.output.extend([out_boxes, out_scores])

        onnx.checker.check_model(model)
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(output_onnx))
        return True
    except Exception as e:
        print(f"[NOTE] Could not prune PP-YOLOE post-processing in ONNX: {e}")
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
            if model_info.architecture == "YOLOv6":
                if not prepare_yolov6_heads(input_path, clean_onnx):
                    shutil.copyfile(input_path, clean_onnx)
            elif model_info.architecture == "PPYOLOE":
                if not prune_ppyoloe_postprocessing(input_path, clean_onnx):
                    shutil.copyfile(input_path, clean_onnx)
            else:
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

        # Generate representative calibration dataset using 500 COCO validation images
        calib_path = work_dir / f"calibration_v3_{model_info.architecture.lower()}_{calib_h}x{calib_w}.npy"
        if not calib_path.exists():
            print(f"Preparing 500-image calibration dataset for {model_info.name}...")
            try:
                import numpy as np
                from PIL import Image
                from .downloader import download_calibration_images

                calib_images = download_calibration_images(output_dir.parent / "downloads", max_images=500)
                samples = []
                for img_p in calib_images:
                    with Image.open(img_p) as im:
                        w, h = im.size
                        scale = min(calib_w / w, calib_h / h)
                        nw, nh = int(round(w * scale)), int(round(h * scale))
                        im_resized = im.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR)

                        # Create gray 114 canvas for letterbox padding
                        canvas = Image.new("RGB", (calib_w, calib_h), (114, 114, 114))
                        pad_x = (calib_w - nw) // 2
                        pad_y = (calib_h - nh) // 2
                        canvas.paste(im_resized, (pad_x, pad_y))

                        arr_rgb = np.array(canvas, dtype=np.float32)

                        if model_info.architecture == "YOLOv6":
                            # YOLOv6 expects RGB normalized to [0.0, 1.0] or [0.0, 255.0]
                            # Using [0.0, 1.0] matches standard YOLOv6 training
                            samples.append(arr_rgb / 255.0)
                        elif model_info.architecture == "PPYOLOE":
                            # PP-YOLOE expects RGB with ImageNet normalization on [0.0, 1.0] scale
                            norm_rgb = arr_rgb / 255.0
                            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                            samples.append((norm_rgb - mean) / std)
                        else:
                            samples.append(arr_rgb / 255.0)

                calib_data = np.stack(samples, axis=0)
                np.save(str(calib_path), calib_data)
                print(f"[OK] Saved {len(samples)} calibration frames ({calib_w}x{calib_h}) to {calib_path.name}")
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
