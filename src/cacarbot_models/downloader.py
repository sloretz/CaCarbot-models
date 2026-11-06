"""Downloads model files with a clear progress bar."""

import os
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm

from .models import ModelInfo


def download_file(url: str, destination: Path) -> Path:
    """Download a file from a URL to destination with progress updates."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"[OK] Already downloaded: {destination.name} ({destination.stat().st_size / (1024*1024):.1f} MB)")
        return destination

    print(f"Downloading {destination.name} from {url}...")
    temp_path = destination.with_suffix(destination.suffix + ".tmp")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 64

        with open(temp_path, "wb") as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    size = f.write(chunk)
                    bar.update(size)

        temp_path.rename(destination)
        print(f"[OK] Saved to {destination}")
        return destination
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def download_model(model_info: ModelInfo, output_dir: Path) -> Path:
    """Download a specific model to the target output directory."""
    target_path = output_dir / model_info.filename
    return download_file(model_info.download_url, target_path)
