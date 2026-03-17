#!/usr/bin/env python3
"""
spatial_graph_to_unity.py - Bridge module to convert spatial graphs to Unity coordinates.

Transforms 2D image-space spatial graphs (from depth_sam3_connector.py) into
3D Unity world coordinates for the embodied evaluation pipeline.

Coordinate Systems
------------------
Spatial Graph (2D image space):
    bbox_center[0]: 0 = left edge, image_width = right edge
    bbox_center[1]: 0 = top edge, image_height = bottom edge
    relative_depth: 0.0 = closest to camera, 1.0 = farthest

Unity (3D world space):
    X: -10 (left) to +10 (right), 0 = center
    Y: 0 (ground) to +10 (up), objects sit at Y = 0.5 (sphere radius)
    Z: -10 (near/foreground) to +10 (far/background)

Usage
-----
    from spatial_graph_to_unity import SpatialGraphToUnity, convert_graph_file

    # Convert a spatial graph JSON file
    unity_entities = convert_graph_file("spatial_outputs/scene_spatial_graph.json")

    # Or use the converter class for more control
    converter = SpatialGraphToUnity()
    unity_entities = converter.convert(graph_data)

    # Use with Unity bridge
    from unity_bridge import UnityBridge
    bridge = UnityBridge()
    for entity in unity_entities:
        bridge.place_object(**entity)
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# Default Unity scene bounds
UNITY_X_RANGE = (-10.0, 10.0)
UNITY_Y_RANGE = (0.0, 10.0)
UNITY_Z_RANGE = (-10.0, 10.0)
UNITY_SPHERE_RADIUS = 0.5


@dataclass
class UnityEntity:
    """Entity with Unity world coordinates."""
    label: str
    x: float
    y: float
    z: float
    color: Optional[str] = None
    scale: float = 1.0
    # Original graph data for reference
    original_id: Optional[str] = None
    z_order: Optional[int] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Unity bridge."""
        d = {
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "scale": self.scale,
        }
        if self.color:
            d["color"] = self.color
        return d


# Category-based height hints (Y coordinate)
# Objects that typically sit on the ground vs elevated
CATEGORY_HEIGHT_MAP = {
    # Ground level (Y = 0.5, sphere sitting on ground)
    "chair": 0.5,
    "table": 0.5,
    "sofa": 0.5,
    "couch": 0.5,
    "bed": 0.5,
    "desk": 0.5,
    "rug": 0.1,
    "carpet": 0.1,
    "ottoman": 0.5,
    "bench": 0.5,
    "stool": 0.5,

    # Slightly elevated (on tables, counters)
    "vase": 1.0,
    "plant": 1.0,
    "cup": 0.8,
    "bowl": 0.8,
    "bottle": 0.8,
    "book": 0.7,
    "laptop": 0.8,
    "monitor": 1.2,
    "tv": 1.5,
    "television": 1.5,

    # Wall-mounted / elevated
    "painting": 2.0,
    "picture": 2.0,
    "mirror": 2.0,
    "clock": 2.5,
    "window": 2.0,
    "cabinet": 1.5,
    "shelf": 1.8,

    # Tall objects
    "lamp": 1.5,
    "floor_lamp": 1.5,
    "standing_lamp": 1.5,
    "refrigerator": 1.5,
    "fridge": 1.5,
    "door": 2.0,
    "curtain": 2.5,

    # Ceiling
    "chandelier": 4.0,
    "ceiling_light": 4.0,
    "fan": 4.0,
}

# Category-based scale hints
CATEGORY_SCALE_MAP = {
    # Large furniture
    "sofa": 2.0,
    "couch": 2.0,
    "bed": 2.0,
    "table": 1.5,
    "desk": 1.5,
    "refrigerator": 1.8,
    "fridge": 1.8,
    "cabinet": 1.5,
    "wardrobe": 1.8,

    # Medium objects
    "chair": 1.0,
    "lamp": 1.0,
    "tv": 1.2,
    "television": 1.2,
    "monitor": 1.0,
    "painting": 1.0,
    "mirror": 1.2,
    "plant": 0.8,

    # Small objects
    "vase": 0.5,
    "cup": 0.3,
    "bowl": 0.4,
    "bottle": 0.4,
    "book": 0.3,
    "clock": 0.4,
}


class SpatialGraphToUnity:
    """
    Converts spatial graph data to Unity world coordinates.

    Parameters
    ----------
    x_range : tuple
        Unity X coordinate range (default: -10 to +10)
    y_range : tuple
        Unity Y coordinate range (default: 0 to +10)
    z_range : tuple
        Unity Z coordinate range (default: -10 to +10)
    use_category_hints : bool
        Use category-based height and scale hints (default: True)
    depth_mapping : str
        How to map depth: "linear" or "logarithmic" (default: "linear")
    min_separation : float
        Minimum separation between objects (default: 1.0)
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = UNITY_X_RANGE,
        y_range: Tuple[float, float] = UNITY_Y_RANGE,
        z_range: Tuple[float, float] = UNITY_Z_RANGE,
        use_category_hints: bool = True,
        depth_mapping: str = "linear",
        min_separation: float = 1.0,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.use_category_hints = use_category_hints
        self.depth_mapping = depth_mapping
        self.min_separation = min_separation

    def convert(
        self,
        graph_data: Dict,
        color_palette: Optional[List[str]] = None,
    ) -> List[UnityEntity]:
        """
        Convert spatial graph to Unity entities.

        Parameters
        ----------
        graph_data : dict
            Spatial graph JSON data (from depth_sam3_connector.py)
        color_palette : list, optional
            List of color names to cycle through

        Returns
        -------
        List[UnityEntity]
            Entities with Unity world coordinates
        """
        nodes = graph_data.get("nodes", [])
        image_size = graph_data.get("image_size", [480, 640])  # [height, width]

        if not nodes:
            return []

        img_h, img_w = image_size

        # Default color palette
        if color_palette is None:
            color_palette = [
                "red", "blue", "green", "yellow", "purple",
                "orange", "cyan", "pink", "brown", "gray"
            ]

        entities = []

        for i, node in enumerate(nodes):
            # Extract fields from spatial graph
            name = node.get("name", node.get("label", f"object_{i}"))
            category = node.get("category", name.split("_")[0])
            bbox_center = node.get("bbox_center", [img_w / 2, img_h / 2])
            bbox = node.get("bbox", [0, 0, img_w, img_h])
            relative_depth = node.get("relative_depth", 0.5)
            z_order = node.get("z_order", i)
            confidence = node.get("confidence", 1.0)
            original_id = node.get("id", f"entity_{i}")

            # Transform coordinates
            unity_x = self._transform_x(bbox_center[0], img_w)
            unity_y = self._compute_y(category, bbox, img_h)
            unity_z = self._transform_z(relative_depth)
            scale = self._compute_scale(category, bbox, img_w, img_h)

            # Assign color
            color = color_palette[i % len(color_palette)]

            entity = UnityEntity(
                label=name,
                x=round(unity_x, 2),
                y=round(unity_y, 2),
                z=round(unity_z, 2),
                color=color,
                scale=round(scale, 2),
                original_id=original_id,
                z_order=z_order,
                confidence=confidence,
            )
            entities.append(entity)

        # Apply separation constraints
        entities = self._enforce_separation(entities)

        return entities

    def _transform_x(self, cx: float, img_w: float) -> float:
        """
        Transform image X coordinate to Unity X.

        Image: 0 = left, img_w = right
        Unity: x_range[0] = left, x_range[1] = right
        """
        # Normalize to [0, 1]
        normalized = cx / img_w
        # Map to Unity range
        x_min, x_max = self.x_range
        return x_min + normalized * (x_max - x_min)

    def _transform_z(self, relative_depth: float) -> float:
        """
        Transform relative depth to Unity Z.

        Depth: 0 = closest (foreground), 1 = farthest (background)
        Unity: z_range[0] = near, z_range[1] = far
        """
        z_min, z_max = self.z_range

        if self.depth_mapping == "logarithmic":
            # Logarithmic mapping: more resolution for nearby objects
            # Add small epsilon to avoid log(0)
            log_depth = math.log1p(relative_depth * 9) / math.log(10)  # log scale 0-1
            return z_min + log_depth * (z_max - z_min)
        else:
            # Linear mapping
            return z_min + relative_depth * (z_max - z_min)

    def _compute_y(self, category: str, bbox: List[float], img_h: float) -> float:
        """
        Compute Unity Y coordinate based on category and image position.

        Uses category hints if available, otherwise infers from
        vertical position in the image.
        """
        if self.use_category_hints:
            # Check category hints (case-insensitive)
            cat_lower = category.lower()
            for cat_key, height in CATEGORY_HEIGHT_MAP.items():
                if cat_key in cat_lower or cat_lower in cat_key:
                    return height

        # Fallback: infer from vertical position in image
        # Objects at the top of the image are likely wall-mounted or elevated
        # Objects at the bottom are likely on the ground
        y1, y2 = bbox[1], bbox[3]
        center_y = (y1 + y2) / 2
        normalized_y = center_y / img_h  # 0 = top, 1 = bottom

        # Invert: top of image → higher Y in Unity
        # Ground objects (bottom of image) → Y near 0.5
        # Elevated objects (top of image) → Y around 2-3
        if normalized_y > 0.7:
            # Bottom of image → ground level
            return UNITY_SPHERE_RADIUS
        elif normalized_y < 0.3:
            # Top of image → wall-mounted / elevated
            return 2.0 + (0.3 - normalized_y) * 5
        else:
            # Middle → slight elevation
            return UNITY_SPHERE_RADIUS + (0.7 - normalized_y) * 2

    def _compute_scale(
        self,
        category: str,
        bbox: List[float],
        img_w: float,
        img_h: float,
    ) -> float:
        """
        Compute Unity scale based on category and bounding box size.
        """
        if self.use_category_hints:
            # Check category hints
            cat_lower = category.lower()
            for cat_key, scale in CATEGORY_SCALE_MAP.items():
                if cat_key in cat_lower or cat_lower in cat_key:
                    return scale

        # Fallback: compute from bbox area
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = img_w * img_h
        area_ratio = bbox_area / image_area

        # Map area ratio to scale
        # Small objects (< 1% of image): scale 0.3-0.5
        # Medium objects (1-5%): scale 0.5-1.0
        # Large objects (5-20%): scale 1.0-2.0
        if area_ratio < 0.01:
            return 0.3 + area_ratio * 20
        elif area_ratio < 0.05:
            return 0.5 + (area_ratio - 0.01) * 12.5
        else:
            return min(2.0, 1.0 + (area_ratio - 0.05) * 6.67)

    def _enforce_separation(self, entities: List[UnityEntity]) -> List[UnityEntity]:
        """
        Adjust positions to enforce minimum separation between objects.

        Uses a simple repulsion approach for overlapping objects.
        """
        if self.min_separation <= 0 or len(entities) < 2:
            return entities

        # Sort by z_order to process front-to-back
        entities = sorted(entities, key=lambda e: e.z_order or 0)

        for i, entity_a in enumerate(entities):
            for entity_b in entities[i + 1:]:
                dx = entity_b.x - entity_a.x
                dz = entity_b.z - entity_a.z
                dist = math.sqrt(dx * dx + dz * dz)

                if dist < self.min_separation and dist > 0:
                    # Push apart
                    overlap = self.min_separation - dist
                    push = overlap / 2 + 0.1

                    # Normalize direction
                    nx, nz = dx / dist, dz / dist

                    entity_a.x -= nx * push
                    entity_a.z -= nz * push
                    entity_b.x += nx * push
                    entity_b.z += nz * push

                    # Clamp to bounds
                    entity_a.x = max(self.x_range[0], min(self.x_range[1], entity_a.x))
                    entity_a.z = max(self.z_range[0], min(self.z_range[1], entity_a.z))
                    entity_b.x = max(self.x_range[0], min(self.x_range[1], entity_b.x))
                    entity_b.z = max(self.z_range[0], min(self.z_range[1], entity_b.z))

        return entities


def convert_graph_file(
    graph_path: str,
    output_path: Optional[str] = None,
    **converter_kwargs,
) -> List[Dict]:
    """
    Convert a spatial graph JSON file to Unity-ready format.

    Parameters
    ----------
    graph_path : str
        Path to spatial graph JSON file
    output_path : str, optional
        Path to save Unity-ready JSON (if provided)
    **converter_kwargs
        Additional arguments for SpatialGraphToUnity

    Returns
    -------
    List[Dict]
        Unity-ready entity dictionaries
    """
    with open(graph_path) as f:
        graph_data = json.load(f)

    converter = SpatialGraphToUnity(**converter_kwargs)
    entities = converter.convert(graph_data)

    # Convert to dicts
    entity_dicts = [asdict(e) for e in entities]

    # Optionally save
    if output_path:
        output = {
            "source_graph": graph_path,
            "image_path": graph_data.get("image_path"),
            "image_size": graph_data.get("image_size"),
            "unity_entities": [e.to_dict() for e in entities],
            "full_entities": entity_dicts,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved Unity-ready graph to: {output_path}")

    return entity_dicts


def convert_for_evaluation(graph_data: Dict) -> Dict:
    """
    Convert spatial graph to format expected by run_unity_eval.py evaluation.

    This bridges the gap between spatial graph structure and the
    evaluation function's expected format.

    Parameters
    ----------
    graph_data : dict
        Spatial graph data (with "nodes" key)

    Returns
    -------
    dict
        Converted data with "entities" key and x/label fields
    """
    nodes = graph_data.get("nodes", [])
    image_size = graph_data.get("image_size", [480, 640])
    img_h, img_w = image_size

    entities = []
    for node in nodes:
        bbox_center = node.get("bbox_center", [img_w / 2, img_h / 2])

        # Normalize x to match Unity scale for comparison
        normalized_x = bbox_center[0] / img_w  # 0-1
        unity_x = (normalized_x - 0.5) * 20     # -10 to +10

        entities.append({
            "label": node.get("name", node.get("label", "")),
            "x": unity_x,  # Now in Unity-comparable scale
            "z_order": node.get("z_order"),
            "relative_depth": node.get("relative_depth"),
            "confidence": node.get("confidence"),
            "original_id": node.get("id"),
        })

    return {
        "entities": entities,
        "image_path": graph_data.get("image_path"),
        "image_size": graph_data.get("image_size"),
    }


# CLI
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert spatial graph to Unity coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert and print
    python spatial_graph_to_unity.py -i spatial_outputs/scene_spatial_graph.json

    # Convert and save
    python spatial_graph_to_unity.py -i spatial_outputs/scene_spatial_graph.json \\
        -o unity_outputs/scene_unity.json

    # Customize mapping
    python spatial_graph_to_unity.py -i graph.json --depth-mapping logarithmic
"""
    )

    parser.add_argument("--input", "-i", required=True, help="Input spatial graph JSON")
    parser.add_argument("--output", "-o", help="Output Unity-ready JSON")
    parser.add_argument("--depth-mapping", choices=["linear", "logarithmic"],
                        default="linear", help="Depth mapping function")
    parser.add_argument("--min-separation", type=float, default=1.0,
                        help="Minimum separation between objects")
    parser.add_argument("--no-category-hints", action="store_true",
                        help="Disable category-based height/scale hints")

    args = parser.parse_args()

    # Convert
    entities = convert_graph_file(
        args.input,
        output_path=args.output,
        depth_mapping=args.depth_mapping,
        min_separation=args.min_separation,
        use_category_hints=not args.no_category_hints,
    )

    # Print summary
    print(f"\nConverted {len(entities)} entities to Unity coordinates:\n")
    print(f"{'Label':<20} {'X':>8} {'Y':>8} {'Z':>8} {'Scale':>8}")
    print("-" * 60)
    for e in entities:
        print(f"{e['label']:<20} {e['x']:>8.2f} {e['y']:>8.2f} {e['z']:>8.2f} {e['scale']:>8.2f}")


if __name__ == "__main__":
    main()
