# How CaCarbot-models Prepares Models for the Coralboard SL NPU

This document describes the compilation pipeline that transforms open vision models into hardware-accelerated binaries for the CaCarbot.


## The Hardware Target: Synaptics Astra SL2610 NPU

The CaCarbot runs on the Synaptics Coralboard SL. The board includes a Neural Processing Unit (NPU) that accelerates computer vision workloads.

The NPU operates under two specific requirements:
1. **8-bit Integer Arithmetic (INT8)**: The NPU processes 8-bit integers rather than 32-bit floating-point numbers.
2. **Virtual Machine FlatBuffer (.vmfb)**: The NPU runtime (`torq-run-module`) loads pre-compiled `.vmfb` program binaries directly through the `/dev/torq` driver.


## The Conversion Pipeline

Standard models from open-source repositories arrive as 32-bit floating-point ONNX files. This tool converts them into `.vmfb` files through five automated steps:

```
Upstream ONNX Model (FP32)
            │
            ▼
[Step 1] Graph Optimization (YOLOv6 Head Decoupling & PP-YOLOE+ Pruning)
            │
            ▼
[Step 2] Full INT8 Quantization (onnx2tf with COCO500 Calibration)
            │
            ▼
[Step 3] TOSA MLIR Conversion (tosa-converter-for-tflite)
            │
            ▼
[Step 4] Hardware Compilation (torq-compile)
            │
            ▼
Coralboard SL .vmfb Binary
```


## Step 1: Graph Optimization

Standard ONNX exports frequently include operations that fail on fixed-point NPUs or degrade quantization precision. This tool automatically repairs these patterns before compilation:

### YOLOv6 Head Decoupling
In standard YOLOv6 ONNX exports, bounding box coordinates, objectness logits, and class probabilities are concatenated into a single output tensor `[1, 8400, 85]`. Forcing bounding box pixel coordinates (hundreds of pixels) and class scores (0.0 to 1.0) into the same INT8 quantization range causes severe rounding errors. This tool decouples the graph into two independent output tensors (`boxes` and `scores`), allowing the compiler to calibrate coordinate scales and class probabilities independently.

### PP-YOLOE+ Post-Processing Pruning
PP-YOLOE+ ONNX graphs contain dynamic Non-Maximum Suppression (NMS), `Shape`, `Slice`, and `Gather` operators. Dynamic-dimension outputs cannot run on fixed-point NPU engines. This tool strips the dynamic post-processing nodes and extracts the raw static feature heads (`boxes` and `scores`), generating clean, fixed-size tensors.


## Step 2: Full INT8 Quantization

The NPU executes math faster and uses less memory when working with 8-bit integers.

This tool:
1. Downloads representative images from the curated 500-image COCO validation subset (covering all 80 classes).
2. Preprocesses the images with aspect-ratio-preserving letterbox padding according to model specifications (RGB in [0, 1] for YOLOv6, ImageNet-normalized RGB for PP-YOLOE+).
3. Feeds these real calibration images through the model.
4. Measures the minimum and maximum signal values across every layer.
5. Generates scale and zero-point parameters.
6. Produces a fully integer-quantized model via `onnx2tf`.


## Step 3: TOSA MLIR Conversion

The Synaptics Torq compiler accepts input programs in the TOSA (Tensor Operator Set Architecture) MLIR format.

This tool runs `tosa-converter-for-tflite` to translate the quantized model into clean, static TOSA operations.


## Step 4: Torq Hardware Compilation

The tool invokes `torq-compile` with the target hardware flag:
```bash
torq-compile model.tosa.mlir -o model.vmfb --torq-hw=SL2610
```

`torq-compile` generates:
* Memory layouts for the SL2610 on-chip SRAM.
* Execution schedules for the Torq compute engine.
* A finalized `.vmfb` binary file ready for deployment.


## Step 5: Deployment to the CaCarbot

You transfer the resulting `.vmfb` binary to the robot using `adb`:
```bash
adb push model.vmfb /home/root/
```

You can then execute inference on the board using `torq-run-module`:
```bash
adb shell "torq-run-module --module=/home/root/model.vmfb --input=1x416x416x3xi8"
```
The driver sends the workload directly to `/dev/torq` and outputs the detections in real time.
