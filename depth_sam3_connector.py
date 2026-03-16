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
        device: str = "cuda",
        # Filtering parameters
        max_instances_per_category: int = 0,  # 0 = no limit
        min_confidence: float = 0.0,  # Additional confidence filter (post-SAM3)
        min_bbox_area_ratio: float = 0.0,  # Min bbox area as ratio of image area
    ):
        self.device = device
        self.depth_backend = depth_backend

        # Filtering settings
        self.max_instances_per_category = max_instances_per_category
        self.min_confidence = min_confidence
        self.min_bbox_area_ratio = min_bbox_area_ratio

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

    def _compute_bbox_area(self, bbox: List[float]) -> float:
        """Compute bounding box area."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _filter_entities(
        self,
        entities: List[Dict],
        image_area: float,
    ) -> List[Dict]:
        """
        Filter entities based on configured thresholds.

        Filtering steps:
        1. Filter by minimum confidence
        2. Filter by minimum bounding box area
        3. Limit instances per category (keep top by confidence)
        """
        filtered = entities

        # Step 1: Filter by confidence
        if self.min_confidence > 0:
            before_count = len(filtered)
            filtered = [e for e in filtered if e["confidence"] >= self.min_confidence]
            if len(filtered) < before_count:
                print(f"    Filtered {before_count - len(filtered)} entities below confidence {self.min_confidence}")

        # Step 2: Filter by minimum bbox area
        if self.min_bbox_area_ratio > 0:
            min_area = image_area * self.min_bbox_area_ratio
            before_count = len(filtered)
            filtered = [e for e in filtered if self._compute_bbox_area(e["bbox"]) >= min_area]
            if len(filtered) < before_count:
                print(f"    Filtered {before_count - len(filtered)} entities below min area ratio {self.min_bbox_area_ratio}")

        # Step 3: Limit instances per category
        if self.max_instances_per_category > 0:
            # Group by category
            by_category: Dict[str, List[Dict]] = {}
            for e in filtered:
                cat = e["category"]
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(e)

            # Keep top N per category by confidence
            limited = []
            for cat, cat_entities in by_category.items():
                # Sort by confidence descending
                cat_entities.sort(key=lambda x: x["confidence"], reverse=True)
                kept = cat_entities[:self.max_instances_per_category]
                if len(cat_entities) > len(kept):
                    print(f"    Limited {cat}: {len(cat_entities)} -> {len(kept)} instances")
                limited.extend(kept)

            filtered = limited

        return filtered

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

        # Apply filtering
        image_area = image.width * image.height
        pre_filter_count = len(raw_entities)
        raw_entities = self._filter_entities(raw_entities, image_area)
        if len(raw_entities) < pre_filter_count:
            print(f"  Filtered: {pre_filter_count} -> {len(raw_entities)} entities")

        # Re-number entities after filtering and reassign names with sequential indices
        by_category: Dict[str, int] = {}
        for e in raw_entities:
            cat = e["category"]
            idx = by_category.get(cat, 0)
            by_category[cat] = idx + 1
            e["name"] = f"{cat}_{idx}" if by_category[cat] > 1 or self.max_instances_per_category > 0 else cat

        for i, e in enumerate(raw_entities):
            e["id"] = f"entity_{i}"

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
  # With inline prompts (manual)
  python depth_sam3_connector.py --image scene.jpg --prompts "table" "chair" "lamp"

  # With prompts file
  python depth_sam3_connector.py --image scene.jpg --prompts-file outputs/prompts.json

  # Auto-detect with GPT-4o
  python depth_sam3_connector.py --image scene.jpg --prompt-source gpt4o

  # Use COCO vocabulary for indoor scenes
  python depth_sam3_connector.py --image scene.jpg --prompt-source vocabulary --scene-type indoor

  # Use COCO ground-truth (for COCO dataset images)
  python depth_sam3_connector.py --image 000000397133.jpg --prompt-source coco_gt \\
      --coco-annotations annotations/instances_val2017.json

  # With filtering
  python depth_sam3_connector.py --image scene.jpg --prompt-source gpt4o \\
      --max_per_category 3 --min_confidence 0.5
"""
    )

    # Input options
    parser.add_argument("--image", "-i", required=True, help="Image path")
    parser.add_argument("--prompts", "-p", nargs="+", help="Text prompts (inline, manual)")
    parser.add_argument("--prompts-file", help="JSON file with prompts")

    # Prompt source options
    parser.add_argument("--prompt-source", choices=["manual", "vocabulary", "coco_gt", "gpt4o"],
                        default="manual", help="Prompt source method (default: manual)")
    parser.add_argument("--scene-type", default="indoor",
                        choices=["all", "indoor", "living_room", "kitchen", "bedroom", "outdoor"],
                        help="Scene type for vocabulary method (default: indoor)")
    parser.add_argument("--coco-annotations", help="Path to COCO instances JSON (for coco_gt)")
    parser.add_argument("--prompt-cache-dir", default="prompt_cache", help="Cache dir for GPT-4o")

    # Output options
    parser.add_argument("--output_dir", "-o", default="spatial_outputs", help="Output directory")
    parser.add_argument("--no_viz", action="store_true", help="Skip visualization")
    parser.add_argument("--no_masks", action="store_true", help="Skip saving masks")
    parser.add_argument("--no_depth", action="store_true", help="Skip saving depth map")

    # Model options
    parser.add_argument("--sam3_bpe", default="sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--sam3_confidence", type=float, default=0.3,
                        help="SAM3 initial confidence threshold (default: 0.3)")
    parser.add_argument("--depth_backend", choices=["dpt", "dinov3"], default="dpt")
    parser.add_argument("--device", default="cuda")

    # Filtering options
    parser.add_argument("--max_per_category", type=int, default=0,
                        help="Max instances per category, 0=unlimited (default: 0)")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="Post-detection confidence filter (default: 0.0)")
    parser.add_argument("--min_area_ratio", type=float, default=0.0,
                        help="Min bbox area as ratio of image (default: 0.0)")

    args = parser.parse_args()

    # Collect prompts based on source
    prompts = []

    if args.prompt_source == "manual":
        # Manual mode: use --prompts and/or --prompts-file
        if args.prompts_file:
            file_prompts = load_prompts_from_file(args.prompts_file)
            prompts.extend(file_prompts)
            print(f"Loaded {len(file_prompts)} prompts from: {args.prompts_file}")

        if args.prompts:
            prompts.extend(args.prompts)

    else:
        # Auto-detection modes
        from prompt_sources import PromptGenerator

        generator = PromptGenerator(
            coco_annotations_path=args.coco_annotations,
            cache_dir=args.prompt_cache_dir,
        )

        if args.prompt_source == "vocabulary":
            prompts = generator.from_coco_vocabulary(
                scene_type=args.scene_type,
                include_extended=True,
            )
            print(f"Using COCO vocabulary ({args.scene_type}): {len(prompts)} prompts")

        elif args.prompt_source == "coco_gt":
            if not args.coco_annotations:
                parser.error("--coco-annotations required for coco_gt prompt source")
            image_filename = Path(args.image).name
            prompts = generator.from_coco_annotations(image_filename=image_filename)
            print(f"Using COCO ground-truth: {len(prompts)} objects in {image_filename}")

        elif args.prompt_source == "gpt4o":
            result = generator.from_gpt4o(args.image)
            prompts = result.get("prompts", [])
            print(f"GPT-4o detected {len(prompts)} objects")
            print(f"  Scene type: {result.get('scene_type')}")
            print(f"  Suggested anchor: {result.get('suggested_anchor')}")

        # Add any additional manual prompts
        if args.prompts:
            prompts.extend(args.prompts)
            print(f"Added {len(args.prompts)} additional manual prompts")

    # Legacy support: if prompts-file provided with non-manual source, merge them
    if args.prompt_source != "manual" and args.prompts_file:
        file_prompts = load_prompts_from_file(args.prompts_file)
        prompts.extend(file_prompts)
        print(f"Merged {len(file_prompts)} prompts from file")

    # Remove duplicates while preserving order
    prompts = list(dict.fromkeys(prompts))

    if not prompts:
        parser.error("Must provide prompts via --prompts, --prompts-file, or --prompt-source")

    # Initialize connector
    connector = DepthSAM3Connector(
        sam3_bpe_path=args.sam3_bpe,
        sam3_confidence=args.sam3_confidence,
        depth_backend=args.depth_backend,
        device=args.device,
        max_instances_per_category=args.max_per_category,
        min_confidence=args.min_confidence,
        min_bbox_area_ratio=args.min_area_ratio,
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
