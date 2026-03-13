from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskType(str, Enum):
    EGOCENTRIC_QA = "egocentric_qa"
    ALLOCENTRIC_QA = "allocentric_qa"


class FrameType(str, Enum):
    VIEWER_CENTERED = "viewer_centered"
    OBJECT_TO_OBJECT = "object_to_object"


class QuerySubtype(str, Enum):
    BINARY_RELATION = "binary_relation"
    RELATION_MCQ = "relation_mcq"
    OBJECT_RETRIEVAL = "object_retrieval"


class Orientation(str, Enum):
    CAMERA = "camera"
    AWAY = "away"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Entity:
    id: str
    name: str
    category: str
    x: float
    y: float
    z_order: int
    relative_depth: float
    bbox: List[float]


@dataclass
class QuerySpec:
    query_id: str
    task_type: TaskType
    frame_type: FrameType
    query_subtype: QuerySubtype
    template_id: str
    prompt: str
    anchor_object: Optional[str]
    reference_object: Optional[str]
    target_object: Optional[str]
    orientation: Optional[Orientation]
    relation_axis: str
    candidate_answers: List[str]
    ground_truth_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["task_type"] = self.task_type.value
        data["frame_type"] = self.frame_type.value
        data["query_subtype"] = self.query_subtype.value
        data["orientation"] = self.orientation.value if self.orientation else None
        return data


def allowed_relation_axes(task_type: TaskType, frame_type: FrameType) -> List[str]:
    if task_type == TaskType.EGOCENTRIC_QA and frame_type == FrameType.VIEWER_CENTERED:
        return ["left_right", "above_below", "foreground_background"]
    if task_type == TaskType.EGOCENTRIC_QA and frame_type == FrameType.OBJECT_TO_OBJECT:
        return ["left_right", "above_below", "front_behind"]
    if task_type == TaskType.ALLOCENTRIC_QA and frame_type in (FrameType.VIEWER_CENTERED, FrameType.OBJECT_TO_OBJECT):
        return ["left_right", "front_behind"]
    return []


def validate_combination(
    task_type: TaskType,
    frame_type: FrameType,
    query_subtype: QuerySubtype,
    relation_axis: str,
) -> None:
    legal_frames = {
        TaskType.EGOCENTRIC_QA: {FrameType.VIEWER_CENTERED, FrameType.OBJECT_TO_OBJECT},
        TaskType.ALLOCENTRIC_QA: {FrameType.VIEWER_CENTERED, FrameType.OBJECT_TO_OBJECT},
    }
    if frame_type not in legal_frames[task_type]:
        raise ValueError(f"Illegal frame '{frame_type.value}' for task '{task_type.value}'")
    if relation_axis not in allowed_relation_axes(task_type, frame_type):
        raise ValueError(
            f"Illegal relation axis '{relation_axis}' for task '{task_type.value}' / frame '{frame_type.value}'"
        )
    if query_subtype not in (
        QuerySubtype.BINARY_RELATION,
        QuerySubtype.RELATION_MCQ,
        QuerySubtype.OBJECT_RETRIEVAL,
    ):
        raise ValueError(f"Unsupported query subtype '{query_subtype.value}'")
