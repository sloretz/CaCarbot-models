# CaCarbot-models

This tool downloads and prepares AI models for the **CaCarbot** on the **Synaptics Coralboard SL** Neural Processing Unit (NPU).

This project uses the [Apache 2.0 License](LICENSE).

## What This Tool Does

The Coralboard SL on the CaCarbot includes a dedicated Neural Processing Unit (NPU). The NPU accelerates computer vision and AI models without loading the main processor.

The NPU executes models in the **`.vmfb`** (Virtual Machine FlatBuffer) binary format.

This tool:
1. Downloads permissively licensed (Apache 2.0) vision models (**Baidu PP-YOLOE+**).
2. Quantizes them to INT8 using representative COCO calibration data and compiles them into `.vmfb` format so you can run them on the board.

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

Or target a specific model (such as **PP-YOLOE+-Small**):
```bash
uv run --extra compile cacarbot-models ppyoloe-plus-s
```

If your platform does not support the compiler wheel or you only want to download model files, omit the extra:
```bash
uv run cacarbot-models
```

The tool stores your models in the `models/` directory.

## Supported Models

| Model Name | License | Input Resolution | Source | Description |
| :--- | :---: | :---: | :--- | :--- |
| **`ppyoloe-plus-s`** | [Apache 2.0](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/LICENSE) | 640×640 | Baidu PaddleDetection | Anchor-free detector with CSPRepResNet backbone and Task Alignment Learning head (~7.9M params). |

## Compiling to `.vmfb`

The tool optimizes, quantizes, and compiles models into `.vmfb` binaries for the Coralboard SL NPU.

- When you pass `--extra compile` on a compatible Linux x86_64 host, `uv` installs the compilation tools and builds the models.
- If your system lacks compiler support or runs another architecture, the tool saves the downloaded models and provides instructions.
- To read the full technical explanation of the conversion pipeline, see [docs/model_pipeline.md](docs/model_pipeline.md).

## Transferring Models to the Coralboard SL

After generating a `.vmfb` file in `models/compiled/`, transfer it to your Coralboard SL via `adb`:

```bash
# Push the model to the root user directory on the board
adb push models/compiled/ppyoloe_plus_s.vmfb /home/root/

# Verify the file on the board
adb shell "ls -lh /home/root/ppyoloe_plus_s.vmfb"
```

You can now run hardware-accelerated inference on the board.
