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
    "yolov6-nano": ModelInfo(
        name="yolov6-nano",
        display_name="YOLOv6-Nano 3.0",
        license="Apache 2.0",
        license_url="https://github.com/meituan/YOLOv6/blob/main/LICENSE",
        architecture="YOLOv6",
        input_resolution="640x640",
        description="High-performance edge detector by Meituan (approx 4.7M parameters) with reparameterized 3x3 Conv backbone and linear DFL regression.",
        download_url="https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6n.onnx",
        filename="yolov6n.onnx",
        is_precompiled_vmfb=False,
    ),
    "yolov6-small": ModelInfo(
        name="yolov6-small",
        display_name="YOLOv6-Small 3.0",
        license="Apache 2.0",
        license_url="https://github.com/meituan/YOLOv6/blob/main/LICENSE",
        architecture="YOLOv6",
        input_resolution="640x640",
        description="High-accuracy real-time object detector by Meituan (approx 18.5M parameters) with RepVGG blocks and linear DFL regression.",
        download_url="https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6s.onnx",
        filename="yolov6s.onnx",
        is_precompiled_vmfb=False,
    ),
    "ppyoloe-plus-s": ModelInfo(
        name="ppyoloe-plus-s",
        display_name="PP-YOLOE+-Small",
        license="Apache 2.0",
        license_url="https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/LICENSE",
        architecture="PPYOLOE",
        input_resolution="640x640",
        description="Anchor-free object detector by Baidu PaddleDetection (approx 7.9M parameters) with CSPRepResNet backbone and TAL head.",
        download_url="https://huggingface.co/chenmohan/web-sdk-pp-detection/resolve/main/ppyoloe-plus-s-640/0.1.0-labs.1/ppyoloe-plus-s-640-fp32.onnx",
        filename="ppyoloe_plus_s.onnx",
        is_precompiled_vmfb=False,
    ),
}
