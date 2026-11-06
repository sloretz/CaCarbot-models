# CaCarbot-models

This tool downloads and prepares AI models for the **CaCarbot** on the **Synaptics Coralboard SL** Neural Processing Unit (NPU).

This project uses the [Apache 2.0 License](LICENSE).

## What This Tool Does

The Coralboard SL on the CaCarbot includes a dedicated Neural Processing Unit (NPU). The NPU accelerates computer vision and AI models without loading the main processor.

The NPU executes models in the **`.vmfb`** (Virtual Machine FlatBuffer) binary format.

This tool:
1. Downloads supported vision models (YOLOX and RTMDet).
2. Compiles them into `.vmfb` format so you can run them on the board.

## Quick Start

You only need Python and [`uv`](https://docs.astral.sh/uv/).

### Step 1: Install uv
Open your terminal and run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: List Available Models
View all supported models and their licenses:
```bash
uv run cacarbot-models --list
```

### Step 3: Download and Prepare Models
To include `torq-compiler` automatically on supported systems, add `--extra compile`:
```bash
uv run --extra compile cacarbot-models
```

Or target a specific model (such as **YOLOX-Nano**):
```bash
uv run --extra compile cacarbot-models yolox-nano
```

If your platform does not support the compiler wheel or you only want to download model files, omit the extra:
```bash
uv run cacarbot-models
```

The tool stores your models in the `models/` directory.

## Supported Models

| Model Name | License | Input Resolution | Source |
| :--- | :---: | :---: | :--- |
| **`yolox-nano`** | [Apache 2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) | 416×416 | Official Megvii ONNX model |
| **`yolox-tiny`** | [Apache 2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) | 416×416 | Official Megvii ONNX model |
| **`rtmdet-nano`** | [Apache 2.0](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE) | 320×320 | Pre-converted RTMDet ONNX model |
| **`rtmdet-tiny`** | [Apache 2.0](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE) | 640×640 | Pre-converted RTMDet ONNX model |

## Compiling to `.vmfb`

The tool optimizes, quantizes, and compiles models into `.vmfb` binaries for the Coralboard SL NPU.

- When you pass `--extra compile` on a compatible Linux x86_64 host, `uv` installs the compilation tools and builds the models.
- If your system lacks compiler support or runs another architecture, the tool saves the downloaded models and provides instructions.
- To read the full technical explanation of the conversion pipeline, see [docs/model_pipeline.md](docs/model_pipeline.md).

## Transferring Models to the Coralboard SL

After generating a `.vmfb` file in `models/compiled/`, transfer it to your Coralboard SL via `adb`:

```bash
# Push the model to the root user directory on the board
adb push models/compiled/yolox-nano.vmfb /home/root/

# Verify the file on the board
adb shell "ls -lh /home/root/*.vmfb"
```

You can now run hardware-accelerated inference on the board.
