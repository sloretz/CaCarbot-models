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
[Step 1] Graph Optimization (Clean Focus Layer)
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


## Step 1: Graph Optimization (Focus Layer)

YOLOX uses a "Focus" layer at its input. This layer downsamples the image by taking alternating pixels and stacking them into channels.

Standard export scripts implement this operation with slicing steps. During conversion, standard tools translate those slicing steps into dynamic slices that exceed memory bounds on fixed-point NPUs.

This tool solves that problem before compilation:
* It reads the ONNX computation graph.
* It removes the slice operations.
* It inserts a single `Reshape` and `Transpose` sequence that performs the exact same mathematical rearrangement.
* The output matches the original model output with zero numerical difference.


## Step 2: Full INT8 Quantization

The NPU executes math faster and uses less memory when working with 8-bit integers.

This tool:
1. Feeds sample calibration images through the model.
2. Measures the minimum and maximum signal values across every layer.
3. Generates scale and zero-point parameters.
4. Produces a fully integer-quantized model via `onnx2tf`.


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
