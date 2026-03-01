#!/usr/bin/env python3
"""
extract_depth.py - Clean depth extraction pipeline

Extracts depth maps from images using either:
- DPT (HuggingFace) - default, easy setup
- DINOv3 - requires local model setup

Usage:
    # Single image with DPT (default)
    python extract_depth.py --image image.jpg --viz

    # Multiple images
    python extract_depth.py --images img1.jpg img2.jpg --output_dir depths/

    # Use DINOv3 backend
    python extract_depth.py --image image.jpg --backend dinov3 --viz
"""

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class DepthResult:
    """Result from depth prediction."""
    depth_map: np.ndarray       # (H, W) depth values
    depth_tensor: torch.Tensor  # Original tensor
    stats: dict                 # min, max, mean, std
    original_size: tuple        # (H, W) original image size


class DPTDepthExtractor:
    """Depth extraction using HuggingFace DPT model (easy setup)."""

    def __init__(
        self,
        model_name: str = "Intel/dpt-large",
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        """Load DPT model from HuggingFace."""
        if self.model is not None:
            return

        print(f"Loading DPT model: {self.model_name}...")
        from transformers import DPTForDepthEstimation, DPTImageProcessor

        self.processor = DPTImageProcessor.from_pretrained(self.model_name)
        self.model = DPTForDepthEstimation.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("DPT model loaded.")

    def extract(self, image: Image.Image) -> DepthResult:
        """Extract depth from PIL Image."""
        self.load()

        original_size = (image.height, image.width)

        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        depth_tensor = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=original_size,
            mode="bicubic",
            align_corners=False
        )

        depth_map = depth_tensor[0, 0].cpu().numpy()

        stats = {
            "min": float(depth_map.min()),
            "max": float(depth_map.max()),
            "mean": float(depth_map.mean()),
            "std": float(depth_map.std()),
            "shape": list(depth_map.shape)
        }

        return DepthResult(
            depth_map=depth_map,
            depth_tensor=depth_tensor.cpu(),
            stats=stats,
            original_size=original_size
        )

    def extract_from_path(self, image_path: str) -> DepthResult:
        """Extract depth from image file."""
        image = Image.open(image_path).convert("RGB")
        return self.extract(image)


class DINOv3DepthExtractor:
    """Depth extraction using DINOv3 model (requires local setup)."""

    DEFAULT_REPO = "./dinov3"
    DEFAULT_BACKBONE = "./dinov3_models/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
    DEFAULT_DEPTH_HEAD = "./dinov3_models/dinov3_vit7b16_synthmix_dpt_head-02040be1.pth"

    def __init__(
        self,
        repo_dir: str = DEFAULT_REPO,
        backbone_weights: str = DEFAULT_BACKBONE,
        depth_head_weights: str = DEFAULT_DEPTH_HEAD,
        img_size: int = 1024,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.repo_dir = repo_dir
        self.backbone_weights = backbone_weights
        self.depth_head_weights = depth_head_weights
        self.img_size = img_size
        self.device = device
        self.dtype = dtype
        self.model = None
        self.transform = None

    def _make_transform(self):
        """Create image transform for depth model."""
        try:
            from torchvision.transforms import v2
            return v2.Compose([
                v2.ToImage(),
                v2.Resize((self.img_size, self.img_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        except ImportError:
            from torchvision import transforms
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def load(self):
        """Load DINOv3 depth model."""
        if self.model is not None:
            return

        print("Loading DINOv3 depth model...")
        print(f"  Backbone: {self.backbone_weights}")
        print(f"  Depth head: {self.depth_head_weights}")

        import sys
        dinov3_path = Path(self.repo_dir)
        if dinov3_path.exists():
            sys.path.insert(0, str(dinov3_path))

        self.model = torch.hub.load(
            self.repo_dir,
            'dinov3_vit7b16_dd',
            source="local",
            weights=self.depth_head_weights,
            backbone_weights=self.backbone_weights
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = self._make_transform()
        print("DINOv3 depth model loaded.")

    def extract(self, image: Image.Image) -> DepthResult:
        """Extract depth from PIL Image."""
        self.load()

        original_size = (image.height, image.width)

        with torch.inference_mode():
            with torch.autocast(self.device, dtype=self.dtype):
                batch = self.transform(image)[None].to(self.device)
                depth_tensor = self.model(batch)

        depth_tensor = depth_tensor.cpu().float()
        depth_map = depth_tensor[0, 0].numpy()

        stats = {
            "min": float(depth_map.min()),
            "max": float(depth_map.max()),
            "mean": float(depth_map.mean()),
            "std": float(depth_map.std()),
            "shape": list(depth_map.shape)
        }

        return DepthResult(
            depth_map=depth_map,
            depth_tensor=depth_tensor,
            stats=stats,
            original_size=original_size
        )

    def extract_from_path(self, image_path: str) -> DepthResult:
        """Extract depth from image file."""
        image = Image.open(image_path).convert("RGB")
        return self.extract(image)


def save_depth(
    result: DepthResult,
    output_path: Path,
    image_name: str,
    original_image: Optional[Image.Image] = None,
    save_tensor: bool = True,
    save_numpy: bool = True,
    save_viz: bool = False
) -> dict:
    """Save depth result to disk."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Save tensor
    if save_tensor:
        tensor_file = output_path / f"{image_name}_depth.pt"
        torch.save(result.depth_tensor, tensor_file)
        saved_files["tensor"] = str(tensor_file)

    # Save numpy
    if save_numpy:
        numpy_file = output_path / f"{image_name}_depth.npy"
        np.save(numpy_file, result.depth_map)
        saved_files["numpy"] = str(numpy_file)

    # Save metadata
    metadata = {
        "image_name": image_name,
        "stats": result.stats,
        "original_size": list(result.original_size),
        "timestamp": datetime.now().isoformat()
    }
    metadata_file = output_path / f"{image_name}_depth_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_files["metadata"] = str(metadata_file)

    # Save visualization
    if save_viz:
        import matplotlib.pyplot as plt

        # Normalize for visualization
        depth_norm = (result.depth_map - result.depth_map.min())
        depth_norm = depth_norm / (depth_norm.max() + 1e-8)

        if original_image is not None:
            # Side-by-side comparison
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            im = axes[1].imshow(depth_norm, cmap='magma')
            axes[1].set_title("Depth Map")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Depth')

            plt.suptitle(f"Depth Extraction: {image_name}", fontsize=14)
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(depth_norm, cmap='magma')
            ax.set_title(f"Depth Map: {image_name}")
            ax.axis('off')
            plt.colorbar(im, fraction=0.046, pad=0.04, label='Depth (normalized)')

        viz_file = output_path / f"{image_name}_depth_viz.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files["viz"] = str(viz_file)

    return saved_files


def find_images(
    directory: str,
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp')
) -> List[str]:
    """Find all images in a directory."""
    dir_path = Path(directory)
    images = []
    for ext in extensions:
        images.extend(dir_path.glob(f"*{ext}"))
        images.extend(dir_path.glob(f"*{ext.upper()}"))
    return sorted([str(p) for p in images])


def main():
    parser = argparse.ArgumentParser(description="Depth Extraction")

    # Input options
    parser.add_argument("--image", "-i", help="Single image path")
    parser.add_argument("--images", nargs="+", help="Multiple image paths")
    parser.add_argument("--input_dir", help="Directory containing images")

    # Output options
    parser.add_argument("--output_dir", "-o", default="depth_outputs", help="Output directory")
    parser.add_argument("--viz", action="store_true", help="Save visualizations")
    parser.add_argument("--no_tensor", action="store_true", help="Skip saving tensors")
    parser.add_argument("--no_numpy", action="store_true", help="Skip saving numpy arrays")

    # Model options
    parser.add_argument("--backend", choices=["dpt", "dinov3"], default="dpt",
                        help="Depth model backend (default: dpt)")
    parser.add_argument("--model_name", default="Intel/dpt-large",
                        help="HuggingFace model name for DPT backend")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    # DINOv3-specific options
    parser.add_argument("--repo_dir", default="./dinov3", help="DINOv3 repo directory")
    parser.add_argument("--backbone", help="DINOv3 backbone weights path")
    parser.add_argument("--depth_head", help="DINOv3 depth head weights path")

    args = parser.parse_args()

    # Collect images
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.images:
        image_paths.extend(args.images)
    if args.input_dir:
        image_paths.extend(find_images(args.input_dir))

    if not image_paths:
        parser.error("No images specified. Use --image, --images, or --input_dir")

    print(f"Found {len(image_paths)} image(s) to process")
    print(f"Backend: {args.backend}")

    # Initialize extractor
    if args.backend == "dpt":
        extractor = DPTDepthExtractor(
            model_name=args.model_name,
            device=args.device
        )
    else:  # dinov3
        extractor_kwargs = {
            "repo_dir": args.repo_dir,
            "device": args.device
        }
        if args.backbone:
            extractor_kwargs["backbone_weights"] = args.backbone
        if args.depth_head:
            extractor_kwargs["depth_head_weights"] = args.depth_head
        extractor = DINOv3DepthExtractor(**extractor_kwargs)

    # Process images
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {image_path}")

        image = Image.open(image_path).convert("RGB")
        result = extractor.extract(image)
        image_name = Path(image_path).stem

        saved = save_depth(
            result, output_dir, image_name,
            original_image=image if args.viz else None,
            save_tensor=not args.no_tensor,
            save_numpy=not args.no_numpy,
            save_viz=args.viz
        )

        print(f"  Stats: min={result.stats['min']:.3f}, max={result.stats['max']:.3f}, "
              f"mean={result.stats['mean']:.3f}")
        print(f"  Saved: {list(saved.keys())}")

    print(f"\nDone! Processed {len(image_paths)} image(s)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
