from __future__ import annotations

from collections import defaultdict
from itertools import product
import random
from typing import DefaultDict, Dict, List, Optional, Tuple

from query_benchmark.ground_truth import (
    compute_ground_truth_answer,
    load_graph,
    parse_entities,
    relation_choices,
)
from query_benchmark.schema import (
    FrameType,
    Orientation,
    QuerySpec,
    QuerySubtype,
    TaskType,
    allowed_relation_axes,
    validate_combination,
)
from query_benchmark.templates import render_prompt


def _template_ids(task_type: TaskType, frame_type: FrameType) -> List[str]:
    del frame_type
    if task_type == TaskType.EGOCENTRIC_QA:
        return ["ego_v1", "ego_v2"]
    return ["allo_v1", "allo_v2"]


def _orientations(task_type: TaskType) -> List[Optional[Orientation]]:
    if task_type == TaskType.ALLOCENTRIC_QA:
        return [Orientation.CAMERA, Orientation.AWAY, Orientation.LEFT, Orientation.RIGHT]
    return [None]


def _make_query(
    query_id: str,
    task_type: TaskType,
    frame_type: FrameType,
    subtype: QuerySubtype,
    template_id: str,
    anchor_object: Optional[str],
    reference_object: Optional[str],
    target_object: Optional[str],
    orientation: Optional[Orientation],
    relation_axis: str,
    candidate_answers: List[str],
    metadata: Dict,
) -> QuerySpec:
    query = QuerySpec(
        query_id=query_id,
        task_type=task_type,
        frame_type=frame_type,
        query_subtype=subtype,
        template_id=template_id,
        prompt="",
        anchor_object=anchor_object,
        reference_object=reference_object,
        target_object=target_object,
        orientation=orientation,
        relation_axis=relation_axis,
        candidate_answers=candidate_answers,
        ground_truth_answer="",
        metadata=metadata,
    )
    query.prompt = render_prompt(query)
    return query


def generate_queries(
    graph_path: str,
    include_object_retrieval: bool = False,
    max_queries: Optional[int] = None,
    queries_per_bucket: int = 10,
    seed: Optional[int] = None,
) -> Dict:
    graph_data = load_graph(graph_path)
    entities = parse_entities(graph_data)
    rng = random.Random(seed)
    by_name = {e.name: e for e in entities}
    bucketed_queries: DefaultDict[Tuple[str, str, str, str], List[QuerySpec]] = defaultdict(list)
    counter = 0

    configs = [
        (TaskType.EGOCENTRIC_QA, FrameType.VIEWER_CENTERED),
        (TaskType.EGOCENTRIC_QA, FrameType.OBJECT_TO_OBJECT),
        (TaskType.ALLOCENTRIC_QA, FrameType.VIEWER_CENTERED),
        (TaskType.ALLOCENTRIC_QA, FrameType.OBJECT_TO_OBJECT),
    ]

    for task_type, frame_type in configs:
        for relation_axis in allowed_relation_axes(task_type, frame_type):
            for subtype in (QuerySubtype.BINARY_RELATION, QuerySubtype.RELATION_MCQ):
                validate_combination(task_type, frame_type, subtype, relation_axis)
                for template_id in _template_ids(task_type, frame_type):
                    for orientation in _orientations(task_type):
                        if frame_type == FrameType.VIEWER_CENTERED:
                            anchors = entities if task_type == TaskType.ALLOCENTRIC_QA else [None]
                            for anchor in anchors:
                                for target in entities:
                                    if anchor is not None and anchor.name == target.name:
                                        continue
                                    choices = relation_choices(task_type, relation_axis)
                                    query = _make_query(
                                        f"q_{counter:05d}",
                                        task_type,
                                        frame_type,
                                        subtype,
                                        template_id,
                                        anchor.name if anchor else None,
                                        None,
                                        target.name,
                                        orientation,
                                        relation_axis,
                                        choices,
                                        {},
                                    )
                                    answer = compute_ground_truth_answer(query, by_name)
                                    if answer in choices:
                                        query.ground_truth_answer = answer
                                        bucket = (task_type.value, frame_type.value, subtype.value, relation_axis)
                                        bucketed_queries[bucket].append(query)
                                        counter += 1
                        else:
                            if task_type == TaskType.EGOCENTRIC_QA:
                                anchors_and_pairs = [(None, reference, target) for reference, target in product(entities, entities)]
                            else:
                                anchors_and_pairs = [
                                    (anchor, reference, target)
                                    for anchor, reference, target in product(entities, entities, entities)
                                ]
                            for anchor, reference, target in anchors_and_pairs:
                                if reference.name == target.name:
                                    continue
                                if anchor is not None and anchor.name in {reference.name, target.name}:
                                    continue
                                choices = relation_choices(task_type, relation_axis)
                                query = _make_query(
                                    f"q_{counter:05d}",
                                    task_type,
                                    frame_type,
                                    subtype,
                                    template_id,
                                    anchor.name if anchor else None,
                                    reference.name,
                                    target.name,
                                    orientation,
                                    relation_axis,
                                    choices,
                                    {},
                                )
                                answer = compute_ground_truth_answer(query, by_name)
                                if answer in choices:
                                    query.ground_truth_answer = answer
                                    bucket = (task_type.value, frame_type.value, subtype.value, relation_axis)
                                    bucketed_queries[bucket].append(query)
                                    counter += 1

            if not include_object_retrieval:
                continue

            subtype = QuerySubtype.OBJECT_RETRIEVAL
            validate_combination(task_type, frame_type, subtype, relation_axis)
            object_names = [e.name for e in entities]
            for template_id in _template_ids(task_type, frame_type):
                for orientation in _orientations(task_type):
                    if frame_type == FrameType.VIEWER_CENTERED:
                        anchors = entities if task_type == TaskType.ALLOCENTRIC_QA else [None]
                        for anchor in anchors:
                            for target_relation in relation_choices(task_type, relation_axis):
                                candidates = object_names
                                if anchor is not None:
                                    candidates = [name for name in object_names if name != anchor.name]
                                query = _make_query(
                                    f"q_{counter:05d}",
                                    task_type,
                                    frame_type,
                                    subtype,
                                    template_id,
                                    anchor.name if anchor else None,
                                    None,
                                    None,
                                    orientation,
                                    relation_axis,
                                    candidates,
                                    {"target_relation": target_relation},
                                )
                                answer = compute_ground_truth_answer(query, by_name)
                                if answer in candidates:
                                    query.ground_truth_answer = answer
                                    bucket = (task_type.value, frame_type.value, subtype.value, relation_axis)
                                    bucketed_queries[bucket].append(query)
                                    counter += 1
                    else:
                        if task_type == TaskType.EGOCENTRIC_QA:
                            anchor_reference_pairs = [(None, reference) for reference in entities]
                        else:
                            anchor_reference_pairs = [
                                (anchor, reference)
                                for anchor, reference in product(entities, entities)
                                if anchor.name != reference.name
                            ]
                        for anchor, reference in anchor_reference_pairs:
                            for target_relation in relation_choices(task_type, relation_axis):
                                candidates = [name for name in object_names if name != reference.name]
                                if anchor is not None:
                                    candidates = [name for name in candidates if name != anchor.name]
                                query = _make_query(
                                    f"q_{counter:05d}",
                                    task_type,
                                    frame_type,
                                    subtype,
                                    template_id,
                                    anchor.name if anchor else None,
                                    reference.name,
                                    None,
                                    orientation,
                                    relation_axis,
                                    candidates,
                                    {"target_relation": target_relation},
                                )
                                answer = compute_ground_truth_answer(query, by_name)
                                if answer in candidates:
                                    query.ground_truth_answer = answer
                                    bucket = (task_type.value, frame_type.value, subtype.value, relation_axis)
                                    bucketed_queries[bucket].append(query)
                                    counter += 1

    queries: List[QuerySpec] = []
    for bucket in sorted(bucketed_queries.keys()):
        bucket_queries = bucketed_queries[bucket]
        if queries_per_bucket > 0 and len(bucket_queries) > queries_per_bucket:
            bucket_queries = rng.sample(bucket_queries, queries_per_bucket)
        bucket_queries.sort(key=lambda q: q.query_id)
        queries.extend(bucket_queries)

    if max_queries is not None:
        queries = queries[:max_queries]

    return {
        "image_path": graph_data.get("image_path"),
        "image_size": graph_data.get("image_size"),
        "metadata": {
            "graph_path": graph_path,
            "num_entities": len(entities),
            "seed": seed,
            "num_queries": len(queries),
            "queries_per_bucket": queries_per_bucket,
            "num_buckets": len(bucketed_queries),
            "include_object_retrieval": include_object_retrieval,
        },
        "queries": [q.to_dict() for q in queries],
    }
