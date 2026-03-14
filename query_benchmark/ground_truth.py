from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from query_benchmark.schema import Entity, FrameType, Orientation, QuerySpec, TaskType


def load_graph(graph_path: str) -> Dict:
    with open(graph_path, "r") as f:
        return json.load(f)


def parse_entities(graph_data: Dict) -> List[Entity]:
    entities: List[Entity] = []
    for node in graph_data.get("nodes", []):
        center = node["bbox_center"]
        entities.append(
            Entity(
                id=str(node["id"]),
                name=str(node["name"]),
                category=str(node.get("category", "unknown")),
                x=float(center[0]),
                y=float(center[1]),
                z_order=int(node["z_order"]),
                relative_depth=float(node.get("relative_depth", 0.0)),
                bbox=[float(v) for v in node["bbox"]],
            )
        )
    entities.sort(key=lambda e: e.z_order)
    return entities


def resolve_image_path(graph_data: Dict, graph_path: str) -> Path:
    raw_path = graph_data.get("image_path")
    if not raw_path:
        raise ValueError("spatial_graph.json is missing image_path")

    image_path = Path(raw_path)
    if image_path.is_absolute() and image_path.exists():
        return image_path

    candidates = [
        (Path(graph_path).parent / raw_path).resolve(),
        (Path.cwd() / raw_path).resolve(),
        (Path.cwd() / "photos" / image_path.name).resolve(),
        (Path.cwd().parent / "photos" / image_path.name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image_path='{raw_path}' from {graph_path}")


def resolve_stage1_labeled_image(graph_path: str) -> Optional[Path]:
    graph_file = Path(graph_path)
    stem = graph_file.stem
    candidates: List[Path] = []

    if stem.endswith("_spatial_graph"):
        prefix = stem[: -len("_spatial_graph")]
        candidates.append(graph_file.with_name(f"{prefix}_spatial_viz.png"))
    candidates.append(graph_file.with_name(f"{stem}_viz.png"))
    candidates.append(graph_file.with_name("benchmark_image.png"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def entities_by_name(entities: List[Entity]) -> Dict[str, Entity]:
    return {e.name: e for e in entities}


def image_center_entity(entities: Dict[str, Entity]) -> Entity:
    center_x = sum(e.x for e in entities.values()) / max(len(entities), 1)
    center_y = sum(e.y for e in entities.values()) / max(len(entities), 1)
    mean_z = sum(e.z_order for e in entities.values()) / max(len(entities), 1)
    return Entity("viewer", "viewer", "viewer", center_x, center_y, int(mean_z), 0.0, [0, 0, 0, 0])


def above_below(reference: Entity, target: Entity, tol: float = 3.0) -> str:
    dy = target.y - reference.y
    if abs(dy) <= tol:
        return "aligned"
    return "below" if dy > 0 else "above"


def left_right(reference: Entity, target: Entity, tol: float = 3.0) -> str:
    dx = target.x - reference.x
    if abs(dx) <= tol:
        return "aligned"
    return "right" if dx > 0 else "left"


def foreground_background(reference: Entity, target: Entity) -> str:
    return "foreground" if target.z_order > reference.z_order else "background"


def get_orientation_basis(orientation: Orientation) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if orientation == Orientation.RIGHT:
        return (1.0, 0.0), (0.0, 1.0)
    if orientation == Orientation.LEFT:
        return (-1.0, 0.0), (0.0, -1.0)
    if orientation == Orientation.CAMERA:
        return (0.0, 1.0), (-1.0, 0.0)
    if orientation == Orientation.AWAY:
        return (0.0, -1.0), (1.0, 0.0)
    return (0.0, 1.0), (-1.0, 0.0)


def project_in_allocentric_frame(anchor: Entity, obj: Entity, orientation: Orientation) -> Tuple[float, float]:
    dx = obj.x - anchor.x
    dz = obj.z_order - anchor.z_order
    forward, right = get_orientation_basis(orientation)
    forward_proj = dx * forward[0] + dz * forward[1]
    right_proj = dx * right[0] + dz * right[1]
    return right_proj, forward_proj


def oriented_front_behind(reference: Entity, target: Entity, orientation: Orientation, tol: float = 1e-6) -> str:
    _, ref_forward = project_in_allocentric_frame(reference, reference, orientation)
    _, target_forward = project_in_allocentric_frame(reference, target, orientation)
    proj = target_forward - ref_forward
    if abs(proj) <= tol:
        return "aligned"
    return "in_front" if proj > 0 else "behind"


def oriented_left_right(reference: Entity, target: Entity, orientation: Orientation, tol: float = 1e-6) -> str:
    ref_right, _ = project_in_allocentric_frame(reference, reference, orientation)
    target_right, _ = project_in_allocentric_frame(reference, target, orientation)
    proj = target_right - ref_right
    if abs(proj) <= tol:
        return "aligned"
    return "right" if proj > 0 else "left"


def relation_choices(task_type: TaskType, relation_axis: str) -> List[str]:
    if relation_axis == "left_right":
        return ["left", "right"]
    if relation_axis == "above_below":
        return ["above", "below"]
    if relation_axis == "foreground_background":
        return ["foreground", "background"]
    if relation_axis == "front_behind":
        return ["in_front", "behind"]
    raise ValueError(f"Unknown relation axis '{relation_axis}'")


def _viewer_centered_answer(query: QuerySpec, entities: Dict[str, Entity]) -> Optional[str]:
    target = entities.get(query.target_object or "")
    if target is None:
        return None

    if query.task_type == TaskType.EGOCENTRIC_QA:
        viewer = image_center_entity(entities)
        if query.relation_axis == "left_right":
            return left_right(viewer, target)
        if query.relation_axis == "above_below":
            return above_below(viewer, target)
        if query.relation_axis == "foreground_background":
            return foreground_background(viewer, target)
        return None

    anchor = entities.get(query.anchor_object or "")
    if anchor is None or query.orientation is None:
        return None
    if query.relation_axis == "left_right":
        right_proj, _ = project_in_allocentric_frame(anchor, target, query.orientation)
        if abs(right_proj) <= 1e-6:
            return "aligned"
        return "right" if right_proj > 0 else "left"
    if query.relation_axis == "front_behind":
        _, forward_proj = project_in_allocentric_frame(anchor, target, query.orientation)
        if abs(forward_proj) <= 1e-6:
            return "aligned"
        return "in_front" if forward_proj > 0 else "behind"
    return None


def _object_to_object_answer(query: QuerySpec, entities: Dict[str, Entity]) -> Optional[str]:
    reference = entities.get(query.reference_object or "")
    target = entities.get(query.target_object or "")
    if reference is None or target is None:
        return None

    if query.task_type == TaskType.EGOCENTRIC_QA:
        if query.relation_axis == "left_right":
            return left_right(reference, target)
        if query.relation_axis == "above_below":
            return above_below(reference, target)
        if query.relation_axis == "front_behind":
            return "in_front" if target.z_order > reference.z_order else "behind"
        return None

    anchor = entities.get(query.anchor_object or "")
    if anchor is None or query.orientation is None:
        return None

    ref_right, ref_forward = project_in_allocentric_frame(anchor, reference, query.orientation)
    target_right, target_forward = project_in_allocentric_frame(anchor, target, query.orientation)

    if query.relation_axis == "left_right":
        delta = target_right - ref_right
        if abs(delta) <= 1e-6:
            return "aligned"
        return "right" if delta > 0 else "left"
    if query.relation_axis == "front_behind":
        delta = target_forward - ref_forward
        if abs(delta) <= 1e-6:
            return "aligned"
        return "in_front" if delta > 0 else "behind"
    return None


def compute_relation_answer(query: QuerySpec, entities: Dict[str, Entity]) -> Optional[str]:
    if query.frame_type == FrameType.VIEWER_CENTERED:
        return _viewer_centered_answer(query, entities)
    return _object_to_object_answer(query, entities)


def compute_ground_truth_answer(query: QuerySpec, entities: Dict[str, Entity]) -> Optional[str]:
    if query.query_subtype.name in ("BINARY_RELATION", "RELATION_MCQ"):
        return compute_relation_answer(query, entities)

    return None


def render_labeled_image(graph_path: str, output_path: str, selected_objects: Optional[List[str]] = None) -> str:
    graph_data = load_graph(graph_path)
    image_path = resolve_image_path(graph_data, graph_path)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()

    selected = set(selected_objects or [])
    nodes = graph_data.get("nodes", [])
    if selected:
        nodes = [node for node in nodes if node["name"] in selected]

    for node in nodes:
        x1, y1, x2, y2 = [int(v) for v in node["bbox"]]
        label = f"z={node['z_order']}: {node['name']}"
        color = (40, 180, 120) if node["z_order"] >= len(nodes) / 2 else (60, 110, 200)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ly = max(0, y1 - th - 6)
        draw.rectangle([x1, ly, x1 + tw + 8, ly + th + 4], fill=(0, 0, 0))
        draw.text((x1 + 4, ly + 2), label, fill=(255, 255, 255), font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path
