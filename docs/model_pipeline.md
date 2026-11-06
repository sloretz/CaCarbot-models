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
[Step 1] Graph Optimization (YOLOX Focus & RTMDet Pruning)
            │
            ▼
[Step 2] Full INT8 Quantization (onnx2tf)
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

Standard ONNX exports frequently include operations that fail on fixed-point NPUs. This tool automatically repairs these patterns before compilation:

### YOLOX Focus Layer Repair
YOLOX uses a "Focus" layer at its input. This layer downsamples the image by taking alternating pixels and stacking them into channels. Standard export scripts implement this operation with slicing steps. Standard conversion tools translate those slicing steps into dynamic slices that exceed memory bounds on fixed-point NPUs.

This tool solves that problem:
* It reads the ONNX computation graph.
* It removes the slice operations.
* It inserts a single `Reshape` and `Transpose` sequence that performs the exact same mathematical rearrangement.
* The output matches the original model output with zero numerical difference.

### Head Decoupling and Graph Optimization
Upstream models often bundle dynamic Non-Maximum Suppression (NMS), `TopK`, and conditional `Where` operations into the ONNX file, or concatenate bounding box coordinates and classification scores into a single output tensor. Combining coordinate magnitudes (in the hundreds) with probability scores (in the range [0, 1]) causes INT8 quantization to crush score precision to zero. Dynamic operators also crash quantization tools and cannot execute on the fixed-point NPU compute engine.

This tool solves these issues:
* **Focus Layer Repair (YOLOX)**: It replaces slicing sequences with a static `Reshape` and `Transpose` sequence that preserves identical mathematical rearrangement.
* **Post-Processing Pruning (RTMDet)**: It extracts raw bounding box coordinates and classification scores, and removes dynamic NMS, `TopK`, and `Where` nodes.
* **Output Head Decoupling (YOLOX and RTMDet)**: It separates predictions into two independent output tensors (`boxes` and `scores`). Independent quantization scales allow the compiler to calibrate coordinate values and score probabilities with full 8-bit precision.


## Step 2: Full INT8 Quantization

The NPU executes math faster and uses less memory when working with 8-bit integers.

This tool:
1. Downloads representative images from the Coco128 dataset.
2. Preprocesses the images with aspect-ratio-preserving letterbox padding according to model specifications (standardized BGR with ImageNet mean/std for RTMDet, raw BGR in [0, 255] for YOLOX).
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
