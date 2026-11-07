"""Downloads model files with a clear progress bar."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tarfile
from typing import List, Optional
import requests
from tqdm import tqdm

from .coco500_manifest import COCO500_IMAGE_FILENAMES
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


def download_calibration_images(cache_dir: Path, max_images: int = 500) -> List[Path]:
    """Download representative images for INT8 quantization calibration.

    Uses a curated set of up to 500 images from COCO val2017 covering all 80 object classes.
    """
    calib_dir = cache_dir / "calibration_images"
    calib_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(calib_dir.glob("*.jpg"))
    if len(existing) >= max_images:
        return existing[:max_images]

    # Check if a pre-existing coco500 tarball is available in cache
    tar_candidates = [
        cache_dir / "coco500.tar.gz",
        cache_dir.parent / "coco500.tar.gz",
        Path("/tmp/coco500.tar.gz"),
    ]
    for tar_p in tar_candidates:
        if tar_p.exists():
            try:
                print(f"Extracting calibration images from {tar_p.name}...")
                with tarfile.open(tar_p, "r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name.endswith(".jpg"):
                            member_file = tar.extractfile(member)
                            if member_file:
                                dest = calib_dir / Path(member.name).name
                                dest.write_bytes(member_file.read())
                existing = sorted(calib_dir.glob("*.jpg"))
                if len(existing) >= max_images:
                    return existing[:max_images]
            except Exception:
                pass

    target_files = COCO500_IMAGE_FILENAMES[:max_images]
    missing = [f for f in target_files if not (calib_dir / f).exists()]

    if missing:
        print(f"Downloading {len(missing)} COCO validation images for INT8 calibration...")
        session = requests.Session()

        def fetch_img(filename: str):
            dest = calib_dir / filename
            if dest.exists() and dest.stat().st_size > 0:
                return
            url = f"http://images.cocodataset.org/val2017/{filename}"
            try:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            except Exception as e:
                print(f"[WARNING] Failed to fetch {filename}: {e}")

        with ThreadPoolExecutor(max_workers=16) as executor:
            list(tqdm(executor.map(fetch_img, missing), total=len(missing), desc="COCO500 calibration"))

    return sorted(calib_dir.glob("*.jpg"))[:max_images]
