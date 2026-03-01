#!/usr/bin/env python3
"""
run_sam3.py - Clean SAM3 segmentation pipeline

Performs text-prompted segmentation using SAM3 model.

Usage:
    # Single image with text prompt
    python run_sam3.py --image image.jpg --prompt "wooden table"

    # Multiple prompts
    python run_sam3.py --image image.jpg --prompts "table" "chair" "lamp"

    # Batch from JSON (VLM results format)
    python run_sam3.py --batch vlm_results.json --output_dir outputs/
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image

# Fix SAM3 import path
sam3_path = Path(__file__).parent / "sam3" / "sam3"
if sam3_path.exists():
    sys.path.insert(0, str(sam3_path.parent))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@dataclass
class SegmentationResult:
    """Result from SAM3 segmentation."""
    prompt: str
    num_objects: int
    masks: np.ndarray          # (N, 1, H, W) binary masks
    boxes: np.ndarray          # (N, 4) xyxy format
    scores: np.ndarray         # (N,) confidence scores
    image_size: tuple          # (H, W)


class SAM3Runner:
    """Clean interface for running SAM3 segmentation."""

    def __init__(
        self,
        bpe_path: str = "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        confidence_threshold: float = 0.3,
        device: str = "cuda"
    ):
        self.bpe_path = bpe_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        """Load SAM3 model."""
        if self.model is not None:
            return

        print("Loading SAM3 model...")

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        self.model = build_sam3_image_model(bpe_path=self.bpe_path)
        self.processor = Sam3Processor(
            self.model,
            confidence_threshold=self.confidence_threshold
        )
        print("SAM3 model loaded.")

    def segment(
        self,
        image: Image.Image,
        prompt: str
    ) -> SegmentationResult:
        """
        Segment image with text prompt.

        Args:
            image: PIL Image
            prompt: Text description of object to segment

        Returns:
            SegmentationResult with masks, boxes, scores
        """
        self.load()

        # Set image and prompt
        state = self.processor.set_image(image)
        state = self.processor.set_text_prompt(prompt=prompt, state=state)

        # Extract results
        masks = state.get("masks", torch.tensor([]))
        boxes = state.get("boxes", torch.tensor([]))
        scores = state.get("scores", torch.tensor([]))

        num_objects = len(scores) if scores.numel() > 0 else 0

        return SegmentationResult(
            prompt=prompt,
            num_objects=num_objects,
            masks=masks.cpu().float().numpy() if num_objects > 0 else np.array([]),
            boxes=boxes.cpu().float().numpy() if num_objects > 0 else np.array([]),
            scores=scores.cpu().float().numpy() if num_objects > 0 else np.array([]),
            image_size=(image.height, image.width)
        )

    def segment_multi(
        self,
        image: Image.Image,
        prompts: List[str]
    ) -> List[SegmentationResult]:
        """
        Segment image with multiple prompts (reuses visual features).

        Args:
            image: PIL Image
            prompts: List of text descriptions

        Returns:
            List of SegmentationResult for each prompt
        """
        self.load()

        # Set image once
        state = self.processor.set_image(image)
        results = []

        for prompt in prompts:
            # Reset prompts but keep visual features
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt=prompt, state=state)

            masks = state.get("masks", torch.tensor([]))
            boxes = state.get("boxes", torch.tensor([]))
            scores = state.get("scores", torch.tensor([]))
            num_objects = len(scores) if scores.numel() > 0 else 0

            results.append(SegmentationResult(
                prompt=prompt,
                num_objects=num_objects,
                masks=masks.cpu().float().numpy() if num_objects > 0 else np.array([]),
                boxes=boxes.cpu().float().numpy() if num_objects > 0 else np.array([]),
                scores=scores.cpu().float().numpy() if num_objects > 0 else np.array([]),
                image_size=(image.height, image.width)
            ))

        return results


def save_results(
    result: SegmentationResult,
    output_dir: Path,
    image_name: str,
    suffix: str = "",
    save_viz: bool = True,
    original_image: Optional[Image.Image] = None
):
    """Save segmentation results to disk."""
    base_name = f"{image_name}{suffix}"

    # Save masks
    masks_file = output_dir / f"{base_name}_masks.npz"
    np.savez_compressed(masks_file, masks=result.masks)

    # Save metadata
    metadata = {
        "prompt": result.prompt,
        "num_objects": result.num_objects,
        "boxes": result.boxes.tolist() if result.num_objects > 0 else [],
        "scores": result.scores.tolist() if result.num_objects > 0 else [],
        "image_size": list(result.image_size),
        "timestamp": datetime.now().isoformat()
    }
    metadata_file = output_dir / f"{base_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save visualization
    if save_viz and original_image is not None:
        import matplotlib.pyplot as plt
        from sam3.visualization_utils import COLORS, plot_mask, plot_bbox

        plt.figure(figsize=(12, 8))
        plt.imshow(original_image)
        plt.title(f"Prompt: {result.prompt}\nDetected: {result.num_objects} object(s)")

        if result.num_objects > 0:
            for i in range(result.num_objects):
                color = COLORS[i % len(COLORS)]
                mask_tensor = torch.from_numpy(result.masks[i]).squeeze(0)
                plot_mask(mask_tensor, color=color)

                h, w = result.image_size
                plot_bbox(
                    h, w,
                    torch.from_numpy(result.boxes[i]),
                    text=f"score={result.scores[i]:.2f}",
                    box_format="XYXY",
                    color=color,
                    relative_coords=False
                )

        plt.axis('off')
        plt.tight_layout()
        viz_file = output_dir / f"{base_name}_viz.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()

    return {"masks": str(masks_file), "metadata": str(metadata_file)}


def process_batch(
    batch_file: str,
    output_dir: str,
    runner: SAM3Runner,
    image_base_dir: str = ".",
    save_viz: bool = True
):
    """Process batch of images from VLM results JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(batch_file, 'r') as f:
        batch_data = json.load(f)

    print(f"Processing {len(batch_data)} images...")

    summary = []
    for item in batch_data:
        image_path = Path(image_base_dir) / item['image_path']
        if not image_path.exists():
            print(f"  Skipping: {image_path} not found")
            continue

        image_name = Path(item['image_file']).stem
        category = item.get('category', '')
        prompts = item.get('descriptions', [])

        if not prompts:
            continue

        # Create category subdirectory
        cat_output = output_path / category if category else output_path
        cat_output.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path)
        results = runner.segment_multi(image, prompts)

        for idx, result in enumerate(results, 1):
            save_results(
                result, cat_output, image_name,
                suffix=f"_desc{idx}",
                save_viz=save_viz,
                original_image=image
            )
            print(f"  {image_name} desc{idx}: {result.num_objects} objects")

        summary.append({
            "image": image_name,
            "prompts": len(prompts),
            "total_objects": sum(r.num_objects for r in results)
        })

    # Save summary
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 Segmentation")

    # Input options
    parser.add_argument("--image", "-i", help="Single image path")
    parser.add_argument("--prompt", "-p", help="Single text prompt")
    parser.add_argument("--prompts", nargs="+", help="Multiple text prompts")
    parser.add_argument("--batch", help="Batch JSON file (VLM results format)")
    parser.add_argument("--image_base_dir", default=".", help="Base dir for batch images")

    # Output options
    parser.add_argument("--output_dir", "-o", default="sam3_outputs", help="Output directory")
    parser.add_argument("--no_viz", action="store_true", help="Skip visualization")

    # Model options
    parser.add_argument("--bpe_path", default="sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Initialize runner
    runner = SAM3Runner(
        bpe_path=args.bpe_path,
        confidence_threshold=args.confidence,
        device=args.device
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        # Batch mode
        process_batch(
            args.batch, args.output_dir, runner,
            image_base_dir=args.image_base_dir,
            save_viz=not args.no_viz
        )
    elif args.image:
        # Single image mode
        image = Image.open(args.image)
        image_name = Path(args.image).stem

        prompts = args.prompts or ([args.prompt] if args.prompt else ["objects"])
        results = runner.segment_multi(image, prompts)

        for idx, result in enumerate(results):
            suffix = f"_{idx}" if len(results) > 1 else ""
            save_results(
                result, output_dir, image_name,
                suffix=suffix,
                save_viz=not args.no_viz,
                original_image=image
            )
            print(f"Prompt '{result.prompt}': {result.num_objects} objects detected")

        print(f"\nResults saved to {output_dir}")
    else:
        parser.error("Provide --image or --batch")


if __name__ == "__main__":
    main()
