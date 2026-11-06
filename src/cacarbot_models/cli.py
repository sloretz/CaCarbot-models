"""Command line interface for CaCarbot-models."""

import argparse
from pathlib import Path
import sys

from .models import MODELS
from .downloader import download_model
from .builder import build_model, check_torq_compiler


def list_models():
    """Print all available models and their details in plain language."""
    print("\nAvailable CaCarbot Models:")
    print("-" * 75)
    for key, info in MODELS.items():
        print(f"- {info.name:15} License: {info.license} ({info.license_url})")
        print(f"  Description: {info.description}")
        print(f"  Input size:  {info.input_resolution}")
        print()
    print("-" * 75)


def run_pipeline(model_names, output_dir: Path, download_only: bool = False):
    """Download and build the selected models."""
    downloads_dir = output_dir / "downloads"
    compiled_dir = output_dir / "compiled"

    output_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    compiled_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTarget directory: {output_dir.resolve()}")
    compiler = check_torq_compiler()
    if compiler:
        print(f"[OK] Found Torq compiler: {compiler}")
    else:
        print("[INFO] Torq compiler not detected in PATH (source models will be downloaded as ONNX).")

    print(f"\nProcessing {len(model_names)} model(s)...")

    results = []
    for name in model_names:
        if name not in MODELS:
            print(f"[WARNING] Unknown model '{name}'. Skipping. (Run 'cacarbot-models list' to see options)")
            continue

        info = MODELS[name]
        print(f"\n--- {info.display_name} ---")

        # 1. Download
        try:
            downloaded_path = download_model(info, downloads_dir)
        except Exception as e:
            print(f"[ERROR] Download failed for {name}: {e}")
            continue

        # 2. Build / Prepare VMFB
        if download_only:
            results.append((name, "Downloaded", downloaded_path))
            continue

        compiled_path = build_model(info, downloaded_path, compiled_dir)
        if compiled_path:
            results.append((name, "Ready (.vmfb)", compiled_path))
        else:
            results.append((name, "Downloaded only (needs torq-compile)", downloaded_path))

    # Summary
    print("\n" + "=" * 75)
    print("Summary of Models:")
    print("=" * 75)
    all_success = True
    for name, status, path in results:
        print(f"- {name:20} : {status}")
        print(f"  File: {path}")
        if not download_only and "Ready (.vmfb)" not in status:
            all_success = False
    print("=" * 75)
    print("\nNext steps to run on your Coralboard SL:")
    print("  adb push <model_file.vmfb> /home/root/")
    print()
    return all_success


def main():
    parser = argparse.ArgumentParser(
        prog="cacarbot-models",
        description="Easy download and build tool for CaCarbot SL AI models (Apache 2.0 and pre-compiled).",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="all",
        help="Model to download and build (e.g. 'yolox-nano', 'yolox-tiny', 'all'). Default: 'all'",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("models"),
        help="Directory to save downloaded and compiled models (default: ./models)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the model weights/ONNX, don't run the Torq compiler.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available models and exit.",
    )

    args = parser.parse_args()

    if args.list or args.model == "list":
        list_models()
        return

    if args.model == "all":
        selected_models = list(MODELS.keys())
    else:
        selected_models = [args.model]

    success = run_pipeline(selected_models, args.output_dir, args.download_only)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
