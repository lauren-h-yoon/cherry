#!/usr/bin/env python3
"""
generate_queries.py

Generate allocentric-style spatial reasoning queries and object prompts for the
Cherry spatial intelligence evaluation pipeline.

This script can work in two modes:

1. FROM IMAGE (--image): Auto-detect objects using GPT-4o, then generate queries
   - Outputs: prompts.json (for SAM3) + queries.json (for evaluation)
   - Use this BEFORE running depth_sam3_connector.py

2. FROM GRAPH (--input): Generate queries from existing spatial_graph.json
   - Outputs: queries.json (for evaluation)
   - Use this AFTER running depth_sam3_connector.py

Features:
- Auto-detect objects in image using GPT-4o (--auto-detect-objects)
- Estimate anchor orientation using GPT-4o
- Cache orientation/detection results to avoid repeated API calls
- Compute ground truth answers from bbox_center, z_order, and estimated orientation
- Use simple occlusion heuristic for visibility

Reference-frame convention:
- above/below use image y relative to the anchor
- left/right/front/behind use an anchor-centered frame derived from estimated orientation

Important z-order convention:
- larger z_order = closer to camera
- smaller z_order = farther from camera
- so z=0 is farthest back, and larger z means more in front

Supported orientation labels from VLM:
- left
- right
- toward_camera
- away_from_camera
- unclear

Usage Examples:
    # Mode 1: Auto-detect objects from image (before SAM3)
    python generate_queries.py --image scene.jpg --auto-detect-objects --anchor "table" -o outputs/

    # Mode 2: Generate queries from existing spatial graph (after SAM3)
    python generate_queries.py --input spatial_graph.json --anchor "table" -o outputs/
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from openai import OpenAI


@dataclass
class Entity:
    id: str
    name: str
    category: str
    x: float
    y: float
    z_order: int
    relative_depth: float
    pixel_count: int
    bbox: List[float]


@dataclass
class QueryExample:
    query_id: str
    query_type: str
    prompt: str
    anchor: str
    targets: List[str]
    answer: Any
    metadata: Dict[str, Any]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def make_orientation_cache_key(image_path: Path, anchor: Entity) -> str:
    raw = f"{image_path.resolve()}|{anchor.id}|{anchor.name}|{anchor.category}|{anchor.bbox}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_cached_orientation(cache_path: Path) -> Optional[Dict[str, Any]]:
    if not cache_path.exists():
        return None
    with open(cache_path, "r") as f:
        return json.load(f)


def save_cached_orientation(cache_path: Path, orientation_info: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(orientation_info, f, indent=2)


def parse_entities(graph_data: Dict[str, Any]) -> List[Entity]:
    if "nodes" not in graph_data or not isinstance(graph_data["nodes"], list):
        raise ValueError("Expected a top-level 'nodes' list in spatial_graph.json")

    entities: List[Entity] = []

    for node in graph_data["nodes"]:
        center = node.get("bbox_center")
        if center is None or len(center) != 2:
            raise ValueError(f"Node {node.get('name', node.get('id'))} is missing bbox_center")

        bbox = node.get("bbox")
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"Node {node.get('name', node.get('id'))} is missing bbox")

        depth_stats = node.get("depth_stats", {})
        pixel_count = int(depth_stats.get("pixel_count", 0))

        entities.append(
            Entity(
                id=str(node["id"]),
                name=str(node["name"]),
                category=str(node["category"]),
                x=float(center[0]),
                y=float(center[1]),
                z_order=int(node["z_order"]),
                relative_depth=float(node.get("relative_depth", 0.0)),
                pixel_count=pixel_count,
                bbox=[float(v) for v in bbox],
            )
        )

    entities.sort(key=lambda e: e.z_order)
    return entities


def get_anchor(
    entities: List[Entity],
    anchor_name: Optional[str],
    anchor_id: Optional[str],
) -> Entity:
    if (anchor_name is None) == (anchor_id is None):
        raise ValueError("Provide exactly one of --anchor or --anchor-id")

    if anchor_name is not None:
        for e in entities:
            if e.name == anchor_name:
                return e
        available = ", ".join(e.name for e in entities)
        raise ValueError(f"Anchor name '{anchor_name}' not found.\nAvailable names: {available}")

    for e in entities:
        if e.id == anchor_id:
            return e

    available = ", ".join(e.id for e in entities)
    raise ValueError(f"Anchor id '{anchor_id}' not found.\nAvailable ids: {available}")


def resolve_image_path(graph_data: Dict[str, Any], input_json_path: str) -> Path:
    if "image_path" not in graph_data:
        raise ValueError("spatial_graph.json is missing image_path")

    raw_path = graph_data["image_path"]
    image_path = Path(raw_path)

    candidates: List[Path] = []

    if image_path.is_absolute():
        candidates.append(image_path)
    else:
        candidates.append((Path(input_json_path).parent / image_path).resolve())
        candidates.append((Path.cwd() / image_path).resolve())
        candidates.append(Path(image_path.name).resolve())
        candidates.append((Path.cwd() / "photos" / image_path.name).resolve())
        candidates.append((Path.cwd().parent / "photos" / image_path.name).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate image file from image_path='{raw_path}'. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_detection_cache_key(image_path: Path) -> str:
    """Create cache key for object detection results."""
    raw = f"{image_path.resolve()}|object_detection"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def detect_objects_in_image(image_path: Path, cache_dir: Path) -> Dict[str, Any]:
    """
    Detect objects in an image using GPT-4o.

    Returns:
        Dict with keys:
        - objects: List of detected object names
        - suggested_anchor: Suggested anchor object
        - confidence: Detection confidence
        - raw_response: Raw model response
    """
    # Check cache first
    cache_key = make_detection_cache_key(image_path)
    cache_path = cache_dir / f"detection_{cache_key}.json"

    if cache_path.exists():
        with open(cache_path, "r") as f:
            cached = json.load(f)
            print(f"  Using cached object detection: {cache_path}")
            return cached

    # Load and encode image
    image = Image.open(image_path).convert("RGB")

    # Resize if too large (to reduce API costs)
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    image_b64 = pil_to_base64(image)

    prompt = """Analyze this image and list all distinct objects you can see.

Return ONLY valid JSON with these keys:
- "objects": List of object names (use simple, common names like "table", "chair", "lamp", "person", "couch", "plant", "tv", "bed", etc.)
- "suggested_anchor": The most prominent/central object that would make a good reference point
- "confidence": A number from 0 to 1 indicating your confidence in the detection
- "scene_type": Brief description of the scene (e.g., "living room", "office", "bedroom")

Guidelines:
- Use singular, lowercase names (e.g., "chair" not "chairs" or "Chair")
- For multiple similar objects, just list the category once (e.g., "chair" not "chair_1, chair_2")
- Include people as "person"
- Be specific but not overly detailed (e.g., "lamp" not "brass floor lamp")
- Only include objects that are clearly visible and distinct

Do not include any text outside the JSON."""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        max_tokens=500,
    )

    content = response.choices[0].message.content
    if content is None:
        # API returned no content (possibly content moderation or error)
        print(f"  Warning: GPT-4o returned no content for image")
        data = {
            "objects": [],
            "suggested_anchor": None,
            "confidence": 0.0,
            "scene_type": "unknown",
            "error": "Model returned no content",
        }
        # Cache empty result
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    text = content.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {
            "objects": [],
            "suggested_anchor": None,
            "confidence": 0.0,
            "scene_type": "unknown",
            "error": f"Could not parse model response: {text[:200]}",
        }

    # Validate and clean up objects list
    if "objects" not in data or not isinstance(data["objects"], list):
        data["objects"] = []

    # Normalize object names
    data["objects"] = [obj.lower().strip() for obj in data["objects"] if obj]
    data["objects"] = list(dict.fromkeys(data["objects"]))  # Remove duplicates, preserve order

    # Store raw response
    data["raw_response"] = text

    # Cache the result
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Cached object detection: {cache_path}")

    return data


def crop_anchor_region(image_path: Path, bbox: List[float], pad: float = 0.2) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    x1 = max(0, x1 - pad * w)
    y1 = max(0, y1 - pad * h)
    x2 = min(image.width, x2 + pad * w)
    y2 = min(image.height, y2 + pad * h)

    return image.crop((x1, y1, x2, y2))


def estimate_anchor_orientation(anchor: Entity, crop: Image.Image) -> Dict[str, Any]:
    client = OpenAI()

    # Resize if too large to reduce API costs and avoid issues
    max_size = 512
    if max(crop.size) > max_size:
        ratio = max_size / max(crop.size)
        new_size = (int(crop.size[0] * ratio), int(crop.size[1] * ratio))
        crop = crop.resize(new_size, Image.LANCZOS)

    image_b64 = pil_to_base64(crop)

    prompt = f"""
You are estimating the facing direction of one object in an indoor scene.

The object is:
- name: {anchor.name}
- category: {anchor.category}

Return ONLY valid JSON with keys:
- "orientation": one of ["left", "right", "toward_camera", "away_from_camera", "unclear"]
- "confidence": a number from 0 to 1
- "reason": a short explanation

Interpret "orientation" as the direction the FRONT of the object is facing in the image.
If the object is too ambiguous, symmetric, or unclear, return "unclear".
Do not include any extra text outside the JSON.
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        max_tokens=250,
    )

    content = response.choices[0].message.content
    if content is None:
        return {
            "orientation": "unclear",
            "confidence": 0.0,
            "reason": "Model returned no content",
        }

    text = content.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {
            "orientation": "unclear",
            "confidence": 0.0,
            "reason": f"Could not parse model response: {text[:200]}",
        }

    allowed = {"left", "right", "toward_camera", "away_from_camera", "unclear"}
    if data.get("orientation") not in allowed:
        data["orientation"] = "unclear"

    try:
        data["confidence"] = float(data.get("confidence", 0.0))
    except Exception:
        data["confidence"] = 0.0

    data["reason"] = str(data.get("reason", ""))

    return data


def heuristic_spatial_distance(anchor: Entity, target: Entity, z_scale: float = 40.0) -> float:
    """
    Heuristic combined distance over image-plane offset and z-order offset.
    """
    dz = (target.z_order - anchor.z_order) * z_scale
    dx = target.x - anchor.x
    dy = target.y - anchor.y
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def above_below(anchor: Entity, target: Entity, tol: float = 3.0) -> str:
    dy = target.y - anchor.y
    if abs(dy) <= tol:
        return "aligned"
    return "below" if dy > 0 else "above"


def bbox_intersection_area(b1: List[float], b2: List[float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def bbox_area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def point_line_distance_2d(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def heuristic_visibility(anchor: Entity, target: Entity, entities: List[Entity]) -> str:
    """
    Simple occlusion heuristic.

    Mark target as occluded if there exists another object that:
    - is between anchor and target in z_order
    - overlaps the target bbox enough OR sits close to the anchor->target line in image plane
    - is not tiny compared to the target
    """
    target_area = bbox_area(target.bbox)
    if target_area <= 0:
        return "visible"

    ax, ay, az = anchor.x, anchor.y, anchor.z_order
    tx, ty, tz = target.x, target.y, target.z_order

    for other in entities:
        if other.id == anchor.id or other.id == target.id:
            continue

        oz = other.z_order

        if not (min(az, tz) < oz < max(az, tz)):
            continue

        inter = bbox_intersection_area(other.bbox, target.bbox)
        other_area = bbox_area(other.bbox)

        overlap_frac_target = inter / target_area if target_area > 0 else 0.0
        overlap_frac_other = inter / other_area if other_area > 0 else 0.0

        dist_to_ray = point_line_distance_2d(other.x, other.y, ax, ay, tx, ty)
        line_len = math.hypot(tx - ax, ty - ay)
        norm_dist = dist_to_ray / max(line_len, 1.0)

        size_ratio = other_area / target_area if target_area > 0 else 0.0

        if (
            overlap_frac_target >= 0.15
            or overlap_frac_other >= 0.15
            or (norm_dist <= 0.12 and size_ratio >= 0.20)
        ):
            return "occluded"

    return "visible"


def get_anchor_basis(orientation: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Returns (forward, right) vectors in the horizontal/depth plane.

    Coordinates:
    - x = image horizontal
    - z = depth, where larger z_order means closer to camera

    forward = direction anchor faces
    right   = anchor's right-hand side
    """
    if orientation == "right":
        forward = (1.0, 0.0)
        right = (0.0, 1.0)
    elif orientation == "left":
        forward = (-1.0, 0.0)
        right = (0.0, -1.0)
    elif orientation == "toward_camera":
        forward = (0.0, 1.0)
        right = (-1.0, 0.0)
    elif orientation == "away_from_camera":
        forward = (0.0, -1.0)
        right = (1.0, 0.0)
    else:
        forward = (0.0, 1.0)
        right = (-1.0, 0.0)

    return forward, right


def oriented_front_behind(anchor: Entity, target: Entity, orientation: str, tol: float = 1e-6) -> str:
    dx = target.x - anchor.x
    dz = target.z_order - anchor.z_order

    forward, _ = get_anchor_basis(orientation)
    proj = dx * forward[0] + dz * forward[1]

    if abs(proj) <= tol:
        return "aligned"
    return "in_front" if proj > 0 else "behind"


def oriented_left_right(anchor: Entity, target: Entity, orientation: str, tol: float = 1e-6) -> str:
    dx = target.x - anchor.x
    dz = target.z_order - anchor.z_order

    _, right = get_anchor_basis(orientation)
    proj = dx * right[0] + dz * right[1]

    if abs(proj) <= tol:
        return "aligned"
    return "right" if proj > 0 else "left"


def nearest_object(anchor: Entity, entities: List[Entity]) -> Entity:
    others = [e for e in entities if e.id != anchor.id]
    return min(others, key=lambda e: heuristic_spatial_distance(anchor, e))


def farthest_object(anchor: Entity, entities: List[Entity]) -> Entity:
    others = [e for e in entities if e.id != anchor.id]
    return max(others, key=lambda e: heuristic_spatial_distance(anchor, e))


def second_closest_object(anchor: Entity, entities: List[Entity]) -> Optional[Entity]:
    others = sorted([e for e in entities if e.id != anchor.id], key=lambda e: heuristic_spatial_distance(anchor, e))
    return others[1] if len(others) >= 2 else None


def closest_in_direction(anchor: Entity, entities: List[Entity], direction: str, orientation: str) -> Optional[Entity]:
    others = [e for e in entities if e.id != anchor.id]

    if direction == "above":
        candidates = [e for e in others if e.y < anchor.y]
    elif direction == "below":
        candidates = [e for e in others if e.y > anchor.y]
    else:
        candidates = []
        for e in others:
            if direction == "left" and oriented_left_right(anchor, e, orientation) == "left":
                candidates.append(e)
            elif direction == "right" and oriented_left_right(anchor, e, orientation) == "right":
                candidates.append(e)
            elif direction == "in_front" and oriented_front_behind(anchor, e, orientation) == "in_front":
                candidates.append(e)
            elif direction == "behind" and oriented_front_behind(anchor, e, orientation) == "behind":
                candidates.append(e)

    if not candidates:
        return None

    return min(candidates, key=lambda e: heuristic_spatial_distance(anchor, e))


def add_query(
    queries: List[QueryExample],
    counter: int,
    query_type: str,
    prompt: str,
    anchor: Entity,
    targets: List[str],
    answer: Any,
    metadata: Dict[str, Any],
) -> int:
    queries.append(
        QueryExample(
            query_id=f"ac_{counter:05d}",
            query_type=query_type,
            prompt=prompt,
            anchor=anchor.name,
            targets=targets,
            answer=answer,
            metadata=metadata,
        )
    )
    return counter + 1


def generate_queries(
    graph_data: Dict[str, Any],
    entities: List[Entity],
    anchor: Entity,
    orientation_info: Dict[str, Any],
) -> Dict[str, Any]:
    queries: List[QueryExample] = []
    counter = 0
    orientation = orientation_info["orientation"]
    orientation_mode = "fallback_unclear" if orientation == "unclear" else "estimated"

    nearest = nearest_object(anchor, entities)
    farthest = farthest_object(anchor, entities)
    second = second_closest_object(anchor, entities)

    counter = add_query(
        queries, counter,
        "nearest_object",
        f"Using the {anchor.name} as the reference point, which object is closest?",
        anchor, [], nearest.name,
        {"distance_metric": "heuristic_spatial_distance"}
    )

    counter = add_query(
        queries, counter,
        "farthest_object",
        f"Using the {anchor.name} as the reference point, which object is farthest away?",
        anchor, [], farthest.name,
        {"distance_metric": "heuristic_spatial_distance"}
    )

    if second is not None:
        counter = add_query(
            queries, counter,
            "second_closest",
            f"Using the {anchor.name} as the reference point, which object is second closest?",
            anchor, [], second.name,
            {"distance_metric": "heuristic_spatial_distance"}
        )

    for target in entities:
        if target.id == anchor.id:
            continue

        counter = add_query(
            queries, counter,
            "left_right",
            f"Using the {anchor.name} as the reference point, is the {target.name} to the left or to the right?",
            anchor, [target.name], oriented_left_right(anchor, target, orientation),
            {"orientation_mode": orientation_mode}
        )

        counter = add_query(
            queries, counter,
            "front_behind",
            f"Using the {anchor.name} as the reference point, is the {target.name} in front of or behind the {anchor.name}?",
            anchor, [target.name], oriented_front_behind(anchor, target, orientation),
            {"orientation_mode": orientation_mode}
        )

        counter = add_query(
            queries, counter,
            "above_below",
            f"Using the {anchor.name} as the reference point, is the {target.name} above or below the {anchor.name}?",
            anchor, [target.name], above_below(anchor, target),
            {}
        )

        visibility_answer = heuristic_visibility(anchor, target, entities)
        counter = add_query(
            queries, counter,
            "visibility",
            f"Using the {anchor.name} as the reference point, is the {target.name} visible or occluded?",
            anchor, [target.name], visibility_answer,
            {"visibility_source": "heuristic_bbox_zorder_occlusion"}
        )

    directions = {
        "left": f"Using the {anchor.name} as the reference point, what is the closest object to the left?",
        "right": f"Using the {anchor.name} as the reference point, what is the closest object to the right?",
        "above": f"Using the {anchor.name} as the reference point, what is the closest object above it?",
        "below": f"Using the {anchor.name} as the reference point, what is the closest object below it?",
        "in_front": f"Using the {anchor.name} as the reference point, what is the closest object in front of it?",
        "behind": f"Using the {anchor.name} as the reference point, what is the closest object behind it?",
    }

    for direction, prompt in directions.items():
        obj = closest_in_direction(anchor, entities, direction, orientation)
        counter = add_query(
            queries, counter,
            f"closest_{direction}",
            prompt,
            anchor, [], obj.name if obj is not None else None,
            {
                "orientation_mode": orientation_mode,
                "distance_metric": "heuristic_spatial_distance"
            }
        )

    return {
        "image_path": graph_data.get("image_path"),
        "image_size": graph_data.get("image_size"),
        "reference_frame": {
            "type": "anchor_centered_estimated_orientation",
            "description": (
                "Queries are evaluated relative to the anchor object's center. "
                "Above/below uses image y. Left/right/front/behind use an anchor-centered "
                "frame derived from VLM-estimated anchor orientation. Visibility uses a simple "
                "bbox/z-order occlusion heuristic."
            ),
        },
        "metadata": {
            "num_entities_total": len(entities),
            "num_queries": len(queries),
            "source_metadata": graph_data.get("metadata", {}),
            "orientation_source": "openai_vision_estimate",
            "orientation_mode": orientation_mode,
            "visibility_source": "heuristic_bbox_zorder_occlusion",
            "distance_metric": "heuristic_spatial_distance",
        },
        "anchor_entity": asdict(anchor),
        "anchor_orientation": orientation_info,
        "queries": [asdict(q) for q in queries],
    }


def make_readable_txt(output_data: Dict[str, Any], output_path: Path) -> None:
    anchor = output_data["anchor_entity"]
    orientation = output_data["anchor_orientation"]
    metadata = output_data["metadata"]
    queries = output_data["queries"]

    lines: List[str] = []
    lines.append("ALLOCENTRIC QUERY SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Image: {output_data.get('image_path')}")
    lines.append("")

    lines.append("ANCHOR OBJECT")
    lines.append("-" * 70)
    lines.append(f"Name: {anchor['name']}")
    lines.append(f"ID: {anchor['id']}")
    lines.append(f"Category: {anchor['category']}")
    lines.append(f"z_order: {anchor['z_order']}")
    lines.append(f"Center: ({anchor['x']:.2f}, {anchor['y']:.2f})")
    lines.append(f"Relative depth: {anchor['relative_depth']:.4f}")
    lines.append(f"Pixel count: {anchor['pixel_count']}")
    lines.append("")

    lines.append("ESTIMATED ANCHOR ORIENTATION")
    lines.append("-" * 70)
    lines.append(f"Orientation: {orientation['orientation']}")
    lines.append(f"Confidence: {orientation['confidence']:.2f}")
    lines.append(f"Reason: {orientation['reason']}")
    lines.append(f"Orientation mode: {metadata['orientation_mode']}")
    lines.append("")

    lines.append("HEURISTICS / SOURCES")
    lines.append("-" * 70)
    lines.append(f"Orientation source: {metadata['orientation_source']}")
    lines.append(f"Visibility source: {metadata['visibility_source']}")
    lines.append(f"Distance metric: {metadata['distance_metric']}")
    lines.append("")

    lines.append("VISIBILITY HEURISTIC")
    lines.append("-" * 70)
    lines.append("A target is marked occluded if another object lies between anchor and target in z-order")
    lines.append("and overlaps the target bbox enough, or lies close to the anchor→target image-plane ray.")
    lines.append("")

    lines.append("Z-ORDER CONVENTION")
    lines.append("-" * 70)
    lines.append("Larger z_order = closer to camera")
    lines.append("Smaller z_order = farther from camera")
    lines.append("For orientation-aware front/behind, anchor orientation is estimated by a VLM.")
    lines.append("")

    def write_section(title: str, section_queries: List[Dict[str, Any]]) -> None:
        if not section_queries:
            return
        lines.append(title)
        lines.append("-" * 70)
        for i, q in enumerate(section_queries, start=1):
            lines.append(f"{i}. Q: {q['prompt']}")
            lines.append(f"   A: {q['answer']}")
            if q.get("targets"):
                lines.append(f"   Targets: {', '.join(q['targets'])}")
            if q.get("metadata"):
                meta_str = ", ".join(f"{k}={v}" for k, v in q["metadata"].items())
                if meta_str:
                    lines.append(f"   Meta: {meta_str}")
            lines.append("")
        lines.append("")

    nearest_queries = [
        q for q in queries
        if q["query_type"] in ["nearest_object", "farthest_object", "second_closest"]
    ]
    lr_queries = [q for q in queries if q["query_type"] == "left_right"]
    depth_queries = [q for q in queries if q["query_type"] == "front_behind"]
    vertical_queries = [q for q in queries if q["query_type"] == "above_below"]
    visibility_queries = [q for q in queries if q["query_type"] == "visibility"]
    directional_queries = [q for q in queries if q["query_type"].startswith("closest_")]

    write_section("NEAREST / FARTHEST OBJECTS", nearest_queries)
    write_section("LEFT / RIGHT RELATION", lr_queries)
    write_section("FRONT / BEHIND RELATION", depth_queries)
    write_section("ABOVE / BELOW RELATION", vertical_queries)
    write_section("VISIBILITY", visibility_queries)
    write_section("DIRECTIONAL NEAREST OBJECT", directional_queries)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def save_prompts_json(objects: List[str], output_path: Path, detection_info: Dict[str, Any]) -> None:
    """Save detected objects as prompts.json for SAM3."""
    prompts_data = {
        "prompts": objects,
        "suggested_anchor": detection_info.get("suggested_anchor"),
        "scene_type": detection_info.get("scene_type", "unknown"),
        "confidence": detection_info.get("confidence", 0.0),
        "num_objects": len(objects),
    }
    with open(output_path, "w") as f:
        json.dump(prompts_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate spatial queries and object prompts for Cherry pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Auto-detect objects from image (before SAM3)
  python generate_queries.py --image scene.jpg --auto-detect-objects -o outputs/

  # Mode 2: Generate queries from existing spatial graph (after SAM3)
  python generate_queries.py --input spatial_graph.json --anchor "table" -o outputs/

  # Combined: Detect objects AND specify anchor
  python generate_queries.py --image scene.jpg --auto-detect-objects --anchor "table" -o outputs/
"""
    )

    # Input options (mutually exclusive modes)
    input_group = parser.add_argument_group("Input (choose one mode)")
    input_group.add_argument("--input", "-i", help="Path to spatial_graph.json (Mode 2: after SAM3)")
    input_group.add_argument("--image", help="Path to image file (Mode 1: before SAM3)")

    # Object detection options
    detect_group = parser.add_argument_group("Object Detection")
    detect_group.add_argument(
        "--auto-detect-objects",
        action="store_true",
        help="Auto-detect objects in image using GPT-4o (requires --image)"
    )
    detect_group.add_argument(
        "--additional-prompts",
        nargs="+",
        help="Additional object prompts to include (e.g., --additional-prompts 'remote' 'book')"
    )

    # Anchor options
    anchor_group = parser.add_argument_group("Anchor Selection")
    anchor_group.add_argument("--anchor", help="Anchor object name (e.g., 'table')")
    anchor_group.add_argument("--anchor-id", help="Anchor object id (e.g., 'entity_1')")
    anchor_group.add_argument(
        "--auto-anchor",
        action="store_true",
        help="Automatically select anchor (uses suggested_anchor from detection or largest object)"
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", "-o", required=True, help="Output directory or query JSON path")

    # Cache options
    parser.add_argument(
        "--cache-dir",
        default="orientation_cache",
        help="Directory for cached orientation/detection estimates"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (always call API)"
    )

    args = parser.parse_args()

    # Validate input mode
    if not args.input and not args.image:
        parser.error("Must provide either --input (spatial_graph.json) or --image")

    if args.auto_detect_objects and not args.image:
        parser.error("--auto-detect-objects requires --image")

    cache_dir = Path(args.cache_dir)

    # Determine output paths
    output_path = Path(args.output)
    if output_path.suffix == ".json":
        # User specified a file path
        output_dir = output_path.parent
        queries_path = output_path
    else:
        # User specified a directory
        output_dir = output_path
        queries_path = output_dir / "queries.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / "prompts.json"

    # ========== MODE 1: From Image (before SAM3) ==========
    if args.image and args.auto_detect_objects:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print("=" * 60)
        print("MODE 1: Object Detection from Image")
        print("=" * 60)
        print(f"Image: {image_path}")

        # Detect objects
        print("\nStep 1: Detecting objects using GPT-4o...")
        detection_info = detect_objects_in_image(image_path, cache_dir)

        objects = detection_info["objects"]
        if args.additional_prompts:
            objects = list(dict.fromkeys(objects + [p.lower().strip() for p in args.additional_prompts]))
            detection_info["objects"] = objects

        print(f"  Detected {len(objects)} objects: {objects}")
        print(f"  Suggested anchor: {detection_info.get('suggested_anchor')}")
        print(f"  Scene type: {detection_info.get('scene_type')}")

        # Save prompts.json
        save_prompts_json(objects, prompts_path, detection_info)
        print(f"\nStep 2: Saved prompts for SAM3: {prompts_path}")

        # If we don't have a spatial graph yet, we can't generate full queries
        # But we can prepare a preliminary query template
        if not args.input:
            # Determine anchor
            anchor_name = args.anchor or detection_info.get("suggested_anchor") or (objects[0] if objects else None)

            preliminary_output = {
                "status": "preliminary",
                "message": "Run depth_sam3_connector.py with prompts.json, then re-run with --input",
                "image_path": str(image_path),
                "detected_objects": objects,
                "anchor": anchor_name,
                "scene_type": detection_info.get("scene_type"),
                "next_steps": [
                    f"python depth_sam3_connector.py --image {image_path} --prompts-file {prompts_path} -o spatial_outputs/",
                    f"python generate_queries.py --input spatial_outputs/*_spatial_graph.json --anchor {anchor_name} -o {output_dir}/"
                ]
            }

            save_json(str(queries_path), preliminary_output)

            print("\n" + "=" * 60)
            print("Preliminary Output Complete")
            print("=" * 60)
            print(f"Prompts file:     {prompts_path}")
            print(f"Preliminary JSON: {queries_path}")
            print(f"\nNext steps:")
            print(f"  1. python depth_sam3_connector.py --image {image_path} --prompts-file {prompts_path} -o spatial_outputs/")
            print(f"  2. python generate_queries.py --input spatial_outputs/*_spatial_graph.json --anchor {anchor_name} -o {output_dir}/")
            print("=" * 60)
            return

    # ========== MODE 2: From Spatial Graph (after SAM3) ==========
    if args.input:
        print("=" * 60)
        print("MODE 2: Query Generation from Spatial Graph")
        print("=" * 60)

        graph_data = load_json(args.input)
        entities = parse_entities(graph_data)

        if len(entities) < 2:
            raise ValueError("Need at least two entities in the graph")

        # Handle anchor selection
        if args.auto_anchor:
            # Use largest object by pixel count, or first object
            anchor = max(entities, key=lambda e: e.pixel_count) if entities else entities[0]
            print(f"Auto-selected anchor: {anchor.name} (largest by pixel count)")
        elif args.anchor or args.anchor_id:
            anchor = get_anchor(entities, args.anchor, args.anchor_id)
        else:
            # List available objects and ask user to specify
            available = ", ".join(e.name for e in entities)
            raise ValueError(f"Must specify --anchor, --anchor-id, or --auto-anchor.\nAvailable: {available}")

        image_path = resolve_image_path(graph_data, args.input)

        cache_key = make_orientation_cache_key(image_path, anchor)
        cache_path = Path(args.cache_dir) / f"{cache_key}.json"

        orientation_info = load_cached_orientation(cache_path)
        if orientation_info is None or args.no_cache:
            print("\nEstimating anchor orientation using GPT-4o...")
            crop = crop_anchor_region(image_path, anchor.bbox)
            orientation_info = estimate_anchor_orientation(anchor, crop)
            save_cached_orientation(cache_path, orientation_info)
            cache_status = f"miss -> saved {cache_path}"
        else:
            cache_status = f"hit -> {cache_path}"

        output = generate_queries(graph_data, entities, anchor, orientation_info)

        save_json(str(queries_path), output)

        stem = queries_path.stem
        txt_path = queries_path.with_name(f"{stem}_readable.txt")
        make_readable_txt(output, txt_path)

        print("\n" + "=" * 60)
        print("Query Generation Complete")
        print("=" * 60)
        print(f"Input graph:      {args.input}")
        print(f"Anchor chosen:    {anchor.name} ({anchor.category}, z={anchor.z_order})")
        print(f"Orientation:      {orientation_info['orientation']} (conf={orientation_info['confidence']:.2f})")
        print(f"Orientation cache:{' '}{cache_status}")
        print(f"JSON output:      {queries_path}")
        print(f"Readable output:  {txt_path}")
        print(f"Queries made:     {output['metadata']['num_queries']}")
        print("=" * 60)


if __name__ == "__main__":
    main()