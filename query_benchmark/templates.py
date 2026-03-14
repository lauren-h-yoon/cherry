from __future__ import annotations

from typing import Dict

from query_benchmark.schema import FrameType, QuerySpec, QuerySubtype, TaskType


EGOCENTRIC_PREFIXES: Dict[str, str] = {
    "ego_v1": "From the viewer's perspective in this image,",
    "ego_v2": "Looking at the scene from the camera's viewpoint,",
}

ALLOCENTRIC_PREFIXES: Dict[str, str] = {
    "allo_v1": "Place yourself at {anchor} and orient yourself to face {orientation_phrase}.",
    "allo_v2": "Imagine you are standing at {anchor}, facing {orientation_phrase}.",
}


def orientation_phrase(query: QuerySpec) -> str:
    mapping = {
        "camera": "the camera",
        "away": "away from the camera",
        "left": "left",
        "right": "right",
    }
    return mapping.get(query.orientation.value if query.orientation else "", "the camera")


def _answer_only_suffix() -> str:
    return "Return only the answer."


def _viewer_prompt(query: QuerySpec, prefix: str) -> str:
    answer_only = _answer_only_suffix()
    if query.query_subtype == QuerySubtype.BINARY_RELATION:
        choices = " or ".join(query.candidate_answers)
        return f"{prefix} is {query.target_object} {choices} relative to you? {answer_only}"
    choices = ", ".join(query.candidate_answers)
    return f"{prefix} where is {query.target_object} relative to you? Choose one: {choices}. {answer_only}"


def _object_to_object_prompt(query: QuerySpec, prefix: str) -> str:
    answer_only = _answer_only_suffix()
    if query.query_subtype == QuerySubtype.BINARY_RELATION:
        choices = " or ".join(query.candidate_answers)
        return f"{prefix} is {query.target_object} {choices} relative to {query.reference_object}? {answer_only}"
    choices = ", ".join(query.candidate_answers)
    return (
        f"{prefix} where is {query.target_object} relative to {query.reference_object}? "
        f"Choose one: {choices}. {answer_only}"
    )


def render_prompt(query: QuerySpec) -> str:
    if query.task_type == TaskType.EGOCENTRIC_QA:
        prefix = EGOCENTRIC_PREFIXES.get(query.template_id, EGOCENTRIC_PREFIXES["ego_v1"])
        if query.frame_type == FrameType.VIEWER_CENTERED:
            return _viewer_prompt(query, prefix)
        return _object_to_object_prompt(query, prefix)

    prefix_template = ALLOCENTRIC_PREFIXES.get(query.template_id, ALLOCENTRIC_PREFIXES["allo_v1"])
    prefix = prefix_template.format(anchor=query.anchor_object, orientation_phrase=orientation_phrase(query))
    if query.frame_type == FrameType.VIEWER_CENTERED:
        return _viewer_prompt(query, prefix)
    return _object_to_object_prompt(query, prefix)
