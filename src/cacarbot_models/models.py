"""Registry of supported AI models for Coralboard SL."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelInfo:
    name: str
    display_name: str
    license: str
    license_url: str
    architecture: str
    input_resolution: str
    description: str
    download_url: str
    filename: str
    is_precompiled_vmfb: bool = False


# Supported models registry (Permissive Apache 2.0 models matching Torq NPU)
MODELS: Dict[str, ModelInfo] = {
    "ppyoloe-plus-s": ModelInfo(
        name="ppyoloe-plus-s",
        display_name="PP-YOLOE+-Small",
        license="Apache 2.0",
        license_url="https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/LICENSE",
        architecture="PPYOLOE",
        input_resolution="640x640",
        description="Anchor-free object detector by Baidu PaddleDetection (approx 7.9M parameters) with CSPRepResNet backbone and Task Alignment Learning (TAL) linear regression head.",
        download_url="https://huggingface.co/chenmohan/web-sdk-pp-detection/resolve/main/ppyoloe-plus-s-640/0.1.0-labs.1/ppyoloe-plus-s-640-fp32.onnx",
        filename="ppyoloe_plus_s.onnx",
        is_precompiled_vmfb=False,
    ),
}
