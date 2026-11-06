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


# Supported models registry
MODELS: Dict[str, ModelInfo] = {
    "yolox-nano": ModelInfo(
        name="yolox-nano",
        display_name="YOLOX-Nano",
        license="Apache 2.0",
        license_url="https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE",
        architecture="YOLOX",
        input_resolution="416x416",
        description="Ultra-lightweight real-time object detector by Megvii (0.91M parameters).",
        download_url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
        filename="yolox_nano.onnx",
        is_precompiled_vmfb=False,
    ),
    "yolox-tiny": ModelInfo(
        name="yolox-tiny",
        display_name="YOLOX-Tiny",
        license="Apache 2.0",
        license_url="https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE",
        architecture="YOLOX",
        input_resolution="416x416",
        description="Fast, higher-accuracy object detector by Megvii (5.06M parameters).",
        download_url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
        filename="yolox_tiny.onnx",
        is_precompiled_vmfb=False,
    ),
}
