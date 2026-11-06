"""Builds and compiles models to .vmfb format for the Coralboard SL NPU."""

import os
from pathlib import Path
import shutil
import subprocess
from typing import Optional

from .models import ModelInfo


def check_torq_compiler() -> Optional[str]:
    """Check if torq-compile is available in the current environment."""
    return shutil.which("torq-compile")


def build_model(model_info: ModelInfo, input_path: Path, output_dir: Path) -> Optional[Path]:
    """Compile an ONNX or raw model into a Coralboard SL .vmfb binary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_vmfb = output_dir / f"{model_info.name}.vmfb"

    # If the downloaded file is already a pre-compiled VMFB, just copy or link it
    if model_info.is_precompiled_vmfb or input_path.suffix == ".vmfb":
        if input_path != target_vmfb:
            shutil.copyfile(input_path, target_vmfb)
        print(f"[OK] Model {model_info.name} is pre-compiled and ready: {target_vmfb}")
        return target_vmfb

    # Check if target .vmfb already exists
    if target_vmfb.exists() and target_vmfb.stat().st_size > 0:
        print(f"[OK] Already compiled: {target_vmfb.name} ({target_vmfb.stat().st_size / (1024*1024):.1f} MB)")
        return target_vmfb

    compiler = check_torq_compiler()
    if not compiler:
        print("\n" + "=" * 70)
        print(f"[INFO] Downloaded {model_info.display_name} successfully to:")
        print(f"   {input_path}")
        print("\n[NOTE] 'torq-compile' was not found on your system PATH.")
        print("   To compile this model into a '.vmfb' binary for the Coralboard SL NPU:")
        print("   1. Run with the optional compile dependency on an x86_64 system:")
        print("      uv run --extra compile cacarbot-models")
        print("   2. Or install torq-compiler manually:")
        print("      pip install torq-compiler")
        print("=" * 70 + "\n")
        return None

    if input_path.suffix.lower() not in [".onnx", ".tosa", ".mlir"]:
        print(f"[NOTE] {model_info.name} is in '{input_path.suffix}' format. torq-compile requires an ONNX file.")
        return None

    print(f"Compiling {model_info.name} using torq-compile for SL2610 NPU...")
    cmd = [
        compiler,
        str(input_path),
        "-o",
        str(target_vmfb),
        "--torq-hw=SL2610",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and target_vmfb.exists() and target_vmfb.stat().st_size > 0:
            print(f"[OK] Successfully compiled {model_info.name} -> {target_vmfb}")
            return target_vmfb
        else:
            print(f"[ERROR] Failed to compile {model_info.name}:")
            if result.stderr:
                print(result.stderr)
            return None
    except Exception as e:
        print(f"[ERROR] Error invoking torq-compile: {e}")
        return None
