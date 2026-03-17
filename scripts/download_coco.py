#!/usr/bin/env python3
"""
download_coco.py - Download COCO 2017 dataset files.

Downloads and extracts COCO dataset components:
  - annotations: Object annotations for train/val 2017
  - train: Training images (118K images, ~18GB)
  - val: Validation images (5K images, ~1GB)

Usage:
    # Download annotations and val images (recommended for dev)
    python scripts/download_coco.py --annotations --val

    # Download everything
    python scripts/download_coco.py --all

    # Download only annotations
    python scripts/download_coco.py --annotations

    # Custom output directory
    python scripts/download_coco.py --val --output ./data/coco
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple

# COCO 2017 URLs
COCO_FILES = {
    "annotations": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "size": "252MB",
        "description": "Train/Val annotations (instances, captions, keypoints)",
    },
    "train": {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "size": "18GB",
        "description": "Training images (118,287 images)",
    },
    "val": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "size": "1GB",
        "description": "Validation images (5,000 images)",
    },
}


def download_with_progress(url: str, dest_path: Path) -> None:
    """Download a file with progress bar."""

    def report_progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:5.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    print(f"  Downloading: {url}")
    print(f"  Destination: {dest_path}")

    urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
    print()  # newline after progress


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file."""
    print(f"  Extracting: {zip_path.name}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"  Extracted to: {extract_to}")


def download_coco_files(
    output_dir: Path,
    download_annotations: bool = False,
    download_train: bool = False,
    download_val: bool = False,
    keep_zip: bool = False,
) -> List[str]:
    """
    Download and extract COCO dataset files.

    Returns list of downloaded components.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    to_download: List[Tuple[str, dict]] = []
    if download_annotations:
        to_download.append(("annotations", COCO_FILES["annotations"]))
    if download_train:
        to_download.append(("train", COCO_FILES["train"]))
    if download_val:
        to_download.append(("val", COCO_FILES["val"]))

    if not to_download:
        print("No files selected for download. Use --annotations, --train, --val, or --all")
        return downloaded

    print(f"\nCOCO 2017 Download")
    print(f"==================")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nFiles to download:")
    for name, info in to_download:
        print(f"  - {name}: {info['description']} ({info['size']})")
    print()

    for name, info in to_download:
        url = info["url"]
        filename = url.split("/")[-1]
        zip_path = output_dir / filename

        print(f"[{name}] {info['description']}")

        # Check if already extracted
        if name == "annotations":
            check_path = output_dir / "annotations"
        else:
            check_path = output_dir / f"{name}2017"

        if check_path.exists():
            print(f"  Already exists: {check_path}")
            downloaded.append(name)
            continue

        # Download if zip doesn't exist
        if not zip_path.exists():
            download_with_progress(url, zip_path)
        else:
            print(f"  Zip already downloaded: {zip_path}")

        # Extract
        extract_zip(zip_path, output_dir)

        # Remove zip if not keeping
        if not keep_zip:
            print(f"  Removing zip: {zip_path.name}")
            zip_path.unlink()

        downloaded.append(name)
        print()

    # Summary
    print("=" * 40)
    print("Download complete!")
    print(f"Location: {output_dir.absolute()}")
    if download_annotations:
        print(f"  annotations/  - COCO annotations JSON files")
    if download_train:
        print(f"  train2017/    - Training images")
    if download_val:
        print(f"  val2017/      - Validation images")

    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download COCO 2017 dataset files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download annotations and validation images (recommended for development)
  python scripts/download_coco.py --annotations --val

  # Download everything
  python scripts/download_coco.py --all

  # Download to custom directory
  python scripts/download_coco.py --val --output ./data/coco

  # Keep zip files after extraction
  python scripts/download_coco.py --val --keep-zip
        """,
    )

    parser.add_argument(
        "--annotations", "-a",
        action="store_true",
        help="Download annotations (252MB)",
    )
    parser.add_argument(
        "--train", "-t",
        action="store_true",
        help="Download training images (18GB, 118K images)",
    )
    parser.add_argument(
        "--val", "-v",
        action="store_true",
        help="Download validation images (1GB, 5K images)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all files (annotations + train + val)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep zip files after extraction",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available files and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available COCO 2017 files:")
        for name, info in COCO_FILES.items():
            print(f"  --{name:12} {info['size']:>6}  {info['description']}")
        return

    if args.all:
        args.annotations = True
        args.train = True
        args.val = True

    download_coco_files(
        output_dir=Path(args.output),
        download_annotations=args.annotations,
        download_train=args.train,
        download_val=args.val,
        keep_zip=args.keep_zip,
    )


if __name__ == "__main__":
    main()
