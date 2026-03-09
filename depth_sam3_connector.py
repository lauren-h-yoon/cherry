#!/usr/bin/env python3
"""
depth_sam3_connector.py - Connector for SAM3 segmentation with depth extraction

Combines SAM3 text-prompted segmentation with depth extraction (DPT/DINOv3)
to produce per-entity depth maps with z-ordering.

Usage:
    # Single image with prompts
    python depth_sam3_connector.py --image scene.jpg --prompts "table" "chair" "lamp"

    # Output includes spatial_graph.json with z-ordered entities
"""

import argparse
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image

# Import our clean modules
from run_sam3 import SAM3Runner, SegmentationResult
from extract_depth import DPTDepthExtractor, DINOv3DepthExtractor, DepthResult


@dataclass
class DepthStats:
    """Depth statistics for a masked region."""
    mean: float
    min: float
    max: float
    std: float
    median: float
    pixel_count: int


@dataclass
class EntityNode:
    """Entity node for spatial graph."""
    id: str
    name: str
    category: str  # The prompt used
    bbox: List[float]  # [x1, y1, x2, y2]
    bbox_center: List[float]  # [cx, cy]
    confidence: float
    z_order: int
    relative_depth: float
    depth_stats: Dict


@dataclass
class SpatialGraph:
    """Spatial graph representation for scene understanding."""
    image_path: str
    image_size: List[int]
    nodes: List[EntityNode]  # Z-ordered entities (index 0 = closest to camera)
    z_order_sequence: List[str]  # Front to back ordering by entity ID
    metadata: Dict


class DepthSAM3Connector:
    """
    Connector that combines SAM3 segmentation with depth extraction.

    Produces z-ordered entities based on depth for graph construction.
    """

    def __init__(
        self,
        sam3_bpe_path: str = "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        sam3_confidence: float = 0.3,
        depth_backend: str = "dpt",  # "dpt" or "dinov3"
        depth_model_name: str = "Intel/dpt-large",
        device: str = "cuda"
    ):
        self.device = device
        self.depth_backend = depth_backend

        # Initialize SAM3
        self.sam3 = SAM3Runner(
            bpe_path=sam3_bpe_path,
            confidence_threshold=sam3_confidence,
            device=device
        )

        # Initialize depth extractor
        if depth_backend == "dpt":
            self.depth_extractor = DPTDepthExtractor(
                model_name=depth_model_name,
                device=device
            )
        else:
            self.depth_extractor = DINOv3DepthExtractor(device=device)

        self._models_loaded = False

    def load_models(self):
        """Load both models."""
        if self._models_loaded:
            return
        print("Loading models...")
        self.sam3.load()
        self.depth_extractor.load()
        self._models_loaded = True
        print("All models loaded.")

    def _compute_depth_stats(self, depth_map: np.ndarray, mask: np.ndarray) -> DepthStats:
        """Compute depth statistics for a masked region."""
        if mask.ndim == 3:
            mask = mask.squeeze()

        # Resize depth map to match mask if needed
        if depth_map.shape != mask.shape:
            depth_pil = Image.fromarray(depth_map.astype(np.float32))
            depth_pil = depth_pil.resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)
            depth_map = np.array(depth_pil)

        masked_depth = depth_map[mask > 0]

        if len(masked_depth) == 0:
            return DepthStats(mean=0, min=0, max=0, std=0, median=0, pixel_count=0)

        return DepthStats(
            mean=float(np.mean(masked_depth)),
            min=float(np.min(masked_depth)),
            max=float(np.max(masked_depth)),
            std=float(np.std(masked_depth)),
            median=float(np.median(masked_depth)),
            pixel_count=int(len(masked_depth))
        )

    def _compute_bbox_center(self, bbox: List[float]) -> List[float]:
        """Compute center of bounding box."""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def analyze(
        self,
        image: Image.Image,
        prompts: List[str],
        image_path: str = ""
    ) -> SpatialGraph:
        """
        Analyze scene and produce z-ordered spatial graph.

        Returns:
            SpatialGraph with z-ordered nodes (entities sorted by depth)
        """
        self.load_models()

        # Run SAM3 segmentation
        print(f"  Running SAM3 with {len(prompts)} prompts...")
        seg_results = self.sam3.segment_multi(image, prompts)

        # Run depth extraction
        print("  Extracting depth...")
        depth_result = self.depth_extractor.extract(image)
        full_depth = depth_result.depth_map

        # Collect all entities with depth info
        raw_entities = []
        entity_id = 0

        for seg in seg_results:
            if seg.num_objects == 0:
                continue

            for obj_idx in range(seg.num_objects):
                mask = seg.masks[obj_idx].squeeze()
                bbox = seg.boxes[obj_idx].tolist()
                score = float(seg.scores[obj_idx])

                depth_stats = self._compute_depth_stats(full_depth, mask)

                raw_entities.append({
                    "id": f"entity_{entity_id}",
                    "name": f"{seg.prompt}_{obj_idx}" if seg.num_objects > 1 else seg.prompt,
                    "category": seg.prompt,
                    "mask": mask,
                    "bbox": bbox,
                    "confidence": score,
                    "depth_stats": depth_stats
                })
                entity_id += 1

        # Sort by median depth and assign z-order
        raw_entities.sort(key=lambda e: e["depth_stats"].median)

        if raw_entities:
            min_depth = raw_entities[0]["depth_stats"].median
            max_depth = raw_entities[-1]["depth_stats"].median
            depth_range = max_depth - min_depth if max_depth > min_depth else 1.0
        else:
            min_depth = max_depth = depth_range = 0

        # Create entity nodes
        entity_nodes = []
        for i, e in enumerate(raw_entities):
            rel_depth = (e["depth_stats"].median - min_depth) / depth_range if depth_range > 0 else 0

            node = EntityNode(
                id=e["id"],
                name=e["name"],
                category=e["category"],
                bbox=e["bbox"],
                bbox_center=self._compute_bbox_center(e["bbox"]),
                confidence=e["confidence"],
                z_order=i,
                relative_depth=rel_depth,
                depth_stats=asdict(e["depth_stats"])
            )
            entity_nodes.append(node)

        # Create spatial graph (z-ordered)
        graph = SpatialGraph(
            image_path=image_path,
            image_size=[image.height, image.width],
            nodes=entity_nodes,
            z_order_sequence=[e.id for e in entity_nodes],
            metadata={
                "num_entities": len(entity_nodes),
                "depth_backend": self.depth_backend,
                "prompts": prompts,
                "timestamp": datetime.now().isoformat()
            }
        )

        # Store masks for later saving (not in graph JSON)
        self._last_masks = {e["id"]: e["mask"] for e in raw_entities}
        self._last_depth = full_depth

        return graph

    def analyze_from_path(self, image_path: str, prompts: List[str]) -> SpatialGraph:
        """Analyze scene from image path."""
        image = Image.open(image_path).convert("RGB")
        return self.analyze(image, prompts, image_path)


def save_spatial_graph(
    graph: SpatialGraph,
    output_dir: Path,
    image_name: str,
    masks: Optional[Dict[str, np.ndarray]] = None,
    depth_map: Optional[np.ndarray] = None,
    save_viz: bool = True,
    original_image: Optional[Image.Image] = None
) -> Dict[str, str]:
    """
    Save spatial graph and associated data.

    Outputs:
    - {image_name}_spatial_graph.json: Complete graph structure for graph conversion
    - {image_name}_masks.npz: All entity masks (optional)
    - {image_name}_depth.npy: Full depth map (optional)
    - {image_name}_viz.png: Visualization (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    # Convert graph to JSON-serializable format
    graph_dict = {
        "image_path": graph.image_path,
        "image_size": graph.image_size,
        "nodes": [
            {
                "id": n.id,
                "name": n.name,
                "category": n.category,
                "bbox": n.bbox,
                "bbox_center": n.bbox_center,
                "confidence": n.confidence,
                "z_order": n.z_order,
                "relative_depth": n.relative_depth,
                "depth_stats": n.depth_stats
            }
            for n in graph.nodes
        ],
        "z_order_sequence": graph.z_order_sequence,
        "metadata": graph.metadata
    }

    # Save spatial graph JSON
    graph_file = output_dir / f"{image_name}_spatial_graph.json"
    with open(graph_file, 'w') as f:
        json.dump(graph_dict, f, indent=2)
    saved_files["spatial_graph"] = str(graph_file)

    # Save masks
    if masks:
        masks_file = output_dir / f"{image_name}_masks.npz"
        np.savez_compressed(masks_file, **{k: v.astype(np.uint8) for k, v in masks.items()})
        saved_files["masks"] = str(masks_file)

    # Save depth map
    if depth_map is not None:
        depth_file = output_dir / f"{image_name}_depth.npy"
        np.save(depth_file, depth_map)
        saved_files["depth"] = str(depth_file)

    # Save visualization
    if save_viz and original_image is not None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: Original with bboxes and z-order labels
        axes[0].imshow(original_image)
        colors = plt.cm.viridis(np.linspace(0, 1, len(graph.nodes)))

        for node, color in zip(graph.nodes, colors):
            x1, y1, x2, y2 = node.bbox
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            axes[0].add_patch(rect)
            axes[0].text(
                x1, y1 - 10,
                f"z={node.z_order}: {node.name}",
                fontsize=8, color='white',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
            )

        axes[0].set_title(f"Detected Entities (z-ordered)\n{len(graph.nodes)} objects")
        axes[0].axis('off')

        # Right: Depth map with entity centers
        if depth_map is not None:
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            # Resize depth to image size if needed
            if depth_norm.shape != (original_image.height, original_image.width):
                depth_pil = Image.fromarray((depth_norm * 255).astype(np.uint8))
                depth_pil = depth_pil.resize((original_image.width, original_image.height), Image.BILINEAR)
                depth_norm = np.array(depth_pil) / 255.0

            axes[1].imshow(depth_norm, cmap='magma')

            for node, color in zip(graph.nodes, colors):
                cx, cy = node.bbox_center
                axes[1].scatter([cx], [cy], c=[color], s=100, edgecolors='white', linewidths=2)
                axes[1].text(cx + 20, cy, f"z={node.z_order}", fontsize=8, color='white')

        axes[1].set_title("Depth Map with Entity Centers")
        axes[1].axis('off')

        plt.suptitle(f"Spatial Analysis: {image_name}", fontsize=14)
        plt.tight_layout()

        viz_file = output_dir / f"{image_name}_spatial_viz.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files["viz"] = str(viz_file)

    return saved_files


def load_prompts_from_file(prompts_file: str) -> List[str]:
    """Load prompts from a JSON file (generated by generate_queries.py)."""
    with open(prompts_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        # Simple list of prompts
        return data
    elif isinstance(data, dict) and "prompts" in data:
        # Output from generate_queries.py
        return data["prompts"]
    else:
        raise ValueError(f"Invalid prompts file format. Expected list or dict with 'prompts' key.")


def main():
    parser = argparse.ArgumentParser(
        description="Depth + SAM3 Spatial Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With inline prompts
  python depth_sam3_connector.py --image scene.jpg --prompts "table" "chair" "lamp"

  # With prompts file from generate_queries.py
  python depth_sam3_connector.py --image scene.jpg --prompts-file outputs/prompts.json

  # Combined (prompts-file + additional prompts)
  python depth_sam3_connector.py --image scene.jpg --prompts-file outputs/prompts.json --prompts "extra_object"
"""
    )

    # Input options
    parser.add_argument("--image", "-i", required=True, help="Image path")
    parser.add_argument("--prompts", "-p", nargs="+", help="Text prompts (inline)")
    parser.add_argument("--prompts-file", help="JSON file with prompts (from generate_queries.py)")

    # Output options
    parser.add_argument("--output_dir", "-o", default="spatial_outputs", help="Output directory")
    parser.add_argument("--no_viz", action="store_true", help="Skip visualization")
    parser.add_argument("--no_masks", action="store_true", help="Skip saving masks")
    parser.add_argument("--no_depth", action="store_true", help="Skip saving depth map")

    # Model options
    parser.add_argument("--sam3_bpe", default="sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--sam3_confidence", type=float, default=0.3)
    parser.add_argument("--depth_backend", choices=["dpt", "dinov3"], default="dpt")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    # Collect prompts from both sources
    prompts = []

    if args.prompts_file:
        file_prompts = load_prompts_from_file(args.prompts_file)
        prompts.extend(file_prompts)
        print(f"Loaded {len(file_prompts)} prompts from: {args.prompts_file}")

    if args.prompts:
        prompts.extend(args.prompts)

    # Remove duplicates while preserving order
    prompts = list(dict.fromkeys(prompts))

    if not prompts:
        parser.error("Must provide prompts via --prompts or --prompts-file")

    # Initialize connector
    connector = DepthSAM3Connector(
        sam3_bpe_path=args.sam3_bpe,
        sam3_confidence=args.sam3_confidence,
        depth_backend=args.depth_backend,
        device=args.device
    )

    # Analyze image
    print(f"Analyzing: {args.image}")
    print(f"Prompts ({len(prompts)}): {prompts}")

    image = Image.open(args.image).convert("RGB")
    graph = connector.analyze(image, prompts, args.image)

    # Save results
    image_name = Path(args.image).stem
    output_dir = Path(args.output_dir)

    saved = save_spatial_graph(
        graph, output_dir, image_name,
        masks=connector._last_masks if not args.no_masks else None,
        depth_map=connector._last_depth if not args.no_depth else None,
        save_viz=not args.no_viz,
        original_image=image
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Z-Order Analysis Complete!")
    print(f"{'='*60}")
    print(f"Entities detected: {len(graph.nodes)}")
    print(f"\nZ-order (front to back):")
    for node in graph.nodes:
        print(f"  z={node.z_order}: {node.name} (depth={node.relative_depth:.3f}, conf={node.confidence:.2f})")

    print(f"\nSaved files:")
    for k, v in saved.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
