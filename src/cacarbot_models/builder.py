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
    """Replace YOLOX slicing Focus layer and decouple box and score output heads."""
    try:
        import numpy as np
        import onnx
        from onnx import helper, TensorProto

        model = onnx.load(str(input_onnx))
        if len(model.graph.node) < 9:
            return False

        # Check if first 8 nodes are Slice and 9th is Concat
        first_ops = [n.op_type for n in model.graph.node[:9]]
        if first_ops == ["Slice"] * 8 + ["Concat"]:
            input_tensor = model.graph.input[0]
            input_dims = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
            if len(input_dims) == 4 and input_dims[2] and input_dims[3]:
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

        # Decouple raw output into separate boxes and scores heads for independent INT8 quantization
        last_node = model.graph.node[-1]
        raw_out = "raw_yolox_output"
        last_node.output[0] = raw_out

        axes_init = helper.make_tensor("slice_axes", TensorProto.INT64, [1], [2])
        starts_box = helper.make_tensor("starts_box", TensorProto.INT64, [1], [0])
        ends_box = helper.make_tensor("ends_box", TensorProto.INT64, [1], [4])
        starts_sc = helper.make_tensor("starts_sc", TensorProto.INT64, [1], [4])
        ends_sc = helper.make_tensor("ends_sc", TensorProto.INT64, [1], [85])
        model.graph.initializer.extend([axes_init, starts_box, ends_box, starts_sc, ends_sc])

        node_box = helper.make_node("Slice", [raw_out, "starts_box", "ends_box", "slice_axes"], ["boxes"], name="SliceBoxes")
        node_sc = helper.make_node("Slice", [raw_out, "starts_sc", "ends_sc", "slice_axes"], ["scores"], name="SliceScores")
        model.graph.node.extend([node_box, node_sc])

        del model.graph.output[:]
        out_b = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [None, 3549, 4])
        out_s = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, 3549, 81])
        model.graph.output.extend([out_b, out_s])

        onnx.checker.check_model(model)
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(output_onnx))
        return True
    except Exception as e:
        print(f"[NOTE] Could not patch Focus layer in ONNX: {e}")
        return False


def prune_rtmdet_postprocessing(input_onnx: Path, output_onnx: Path) -> bool:
    """Prune dynamic NMS and post-processing layers from RTMDet and decouple box and score heads."""
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

        # Find the 4-input coordinate Concat node before NMS (assembles bounding boxes)
        bbox_concat_idx = None
        for i in range(nms_idx - 1, -1, -1):
            n = model.graph.node[i]
            if n.op_type == "Concat" and len(n.input) == 4:
                bbox_concat_idx = i
                break
        if bbox_concat_idx is None:
            return False

        bbox_tensor = model.graph.node[bbox_concat_idx].output[0]

        # Find the last Sigmoid node before NMS (computes class scores)
        scores_tensor = None
        for i in range(nms_idx - 1, -1, -1):
            n = model.graph.node[i]
            if n.op_type == "Sigmoid":
                scores_tensor = n.output[0]
                break
        if scores_tensor is None:
            return False

        # Keep nodes up to the bounding box concat node
        del model.graph.node[bbox_concat_idx + 1:]

        del model.graph.output[:]
        out_boxes = helper.make_tensor_value_info(bbox_tensor, TensorProto.FLOAT, [None, None, 4])
        out_scores = helper.make_tensor_value_info(scores_tensor, TensorProto.FLOAT, [None, None, None])
        model.graph.output.extend([out_boxes, out_scores])

        onnx.checker.check_model(model)
        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(output_onnx))
        return True
    except Exception as e:
        print(f"[NOTE] Could not prune RTMDet post-processing in ONNX: {e}")
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
            if model_info.architecture == "YOLOX":
                if not patch_yolox_focus_layer(input_path, clean_onnx):
                    shutil.copyfile(input_path, clean_onnx)
            elif model_info.architecture == "RTMDet":
                if not prune_rtmdet_postprocessing(input_path, clean_onnx):
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

        # Generate representative calibration dataset
        calib_path = work_dir / f"calibration_v2_{model_info.architecture.lower()}_{calib_h}x{calib_w}.npy"
        if not calib_path.exists():
            print(f"Preparing real calibration dataset for {model_info.name}...")
            try:
                import numpy as np
                from PIL import Image
                from .downloader import download_calibration_images

                calib_images = download_calibration_images(output_dir.parent / "downloads", max_images=32)
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

                        arr = np.array(canvas, dtype=np.float32)
                        arr_bgr = arr[:, :, ::-1].copy()

                        if model_info.architecture == "YOLOX":
                            # YOLOX expects BGR in range [0.0, 255.0]
                            samples.append(arr_bgr)
                        else:
                            # RTMDet from MMDetection expects BGR with ImageNet normalization:
                            # mean = [103.53, 116.28, 123.675], std = [57.375, 57.12, 58.395]
                            mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
                            std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
                            norm_bgr = (arr_bgr - mean) / std
                            samples.append(norm_bgr)

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
