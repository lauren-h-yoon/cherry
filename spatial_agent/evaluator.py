#!/usr/bin/env python3
"""
evaluator.py - Evaluate VLM spatial outputs against ground truth

Compares VLM's outputs (paths, points, regions) from primitive tools
against ground truth from SAM3+Depth pipeline.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

from .ground_truth import GroundTruth, ObjectInfo
from .primitive_tools import RecordedOutputs


class EvaluationMetric(Enum):
    """Types of evaluation metrics."""
    POINTING_ACCURACY = "pointing_accuracy"
    PATH_VALIDITY = "path_validity"
    PATH_EFFICIENCY = "path_efficiency"
    PATH_DIRECTION = "path_direction"
    REGION_IOU = "region_iou"
    SPATIAL_RELATION = "spatial_relation"


@dataclass
class PathDirectionResult:
    """Result of evaluating path directionality."""
    direction_correct: bool
    expected_direction: str  # "left", "right", "forward", "backward", "nearest"
    actual_direction: str    # computed from waypoints
    consistency_score: float  # 0-1, how consistent is movement
    backtrack_count: int     # number of direction reversals

    @property
    def score(self) -> float:
        """Direction score (0-1)."""
        if self.direction_correct:
            return 0.7 + (0.3 * self.consistency_score)
        return 0.3 * self.consistency_score


@dataclass
class PointingResult:
    """Result of evaluating a pointing action."""
    correct: bool
    target_object: str
    vlm_point: Tuple[int, int]
    gt_position: Tuple[float, float]
    distance_error: float
    threshold: float
    # Partial credit fields
    hit_same_category: bool = False  # Hit different object of same category
    hit_object: Optional[str] = None  # Object that was actually hit (if any)
    target_category: Optional[str] = None  # Category of target object

    @property
    def score(self) -> float:
        """
        Score with partial credit:
        - 1.0: Hit exact target object
        - 0.5: Hit different object of same category
        - 0.0: Missed entirely
        """
        if self.correct:
            return 1.0
        elif self.hit_same_category:
            return 0.5
        return 0.0


@dataclass
class PathResult:
    """Result of evaluating a path."""
    start_correct: bool
    end_correct: bool
    path_valid: bool
    collisions: List[str]
    efficiency: float
    vlm_path: List[Tuple[int, int]]
    vlm_path_length: float
    optimal_length: float
    direction_result: Optional[PathDirectionResult] = None

    @property
    def score(self) -> float:
        """Overall path score (0-1)."""
        if self.direction_result is not None:
            # New scoring with direction
            # 20% start, 20% end, 20% valid, 20% efficiency, 20% direction
            score = 0.0
            if self.start_correct:
                score += 0.20
            if self.end_correct:
                score += 0.20
            if self.path_valid:
                score += 0.20
            score += 0.20 * min(self.efficiency, 1.0)
            score += 0.20 * self.direction_result.score
            return score
        else:
            # Legacy scoring without direction
            score = 0.0
            if self.start_correct:
                score += 0.25
            if self.end_correct:
                score += 0.25
            if self.path_valid:
                score += 0.25
            score += 0.25 * min(self.efficiency, 1.0)
            return score


@dataclass
class RegionResult:
    """Result of evaluating a marked region."""
    correct_object: bool
    target_object: str
    vlm_bbox: List[int]
    gt_bbox: List[int]
    iou: float
    center_distance: float

    @property
    def score(self) -> float:
        return self.iou if self.correct_object else 0.0


@dataclass
class SpatialRelationResult:
    """Result of evaluating a spatial relation answer."""
    correct: bool
    relation_type: str
    entity_a: str
    entity_b: str
    vlm_answer: bool
    gt_answer: bool

    @property
    def score(self) -> float:
        return 1.0 if self.correct else 0.0


@dataclass
class YesNoResult:
    """Result of evaluating a yes/no answer."""
    answer_found: bool
    vlm_answer: Optional[bool]  # True=yes, False=no, None=couldn't parse
    gt_answer: bool
    correct: bool

    @property
    def score(self) -> float:
        if not self.answer_found:
            return 0.0
        return 1.0 if self.correct else 0.0


@dataclass
class EvaluationReport:
    """Complete evaluation report for a task."""
    task_id: str
    task_type: str
    pointing_results: List[PointingResult] = field(default_factory=list)
    path_results: List[PathResult] = field(default_factory=list)
    region_results: List[RegionResult] = field(default_factory=list)
    relation_results: List[SpatialRelationResult] = field(default_factory=list)
    yes_no_results: List[YesNoResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Compute overall score across all evaluations."""
        all_scores = []
        all_scores.extend([r.score for r in self.pointing_results])
        all_scores.extend([r.score for r in self.path_results])
        all_scores.extend([r.score for r in self.region_results])
        all_scores.extend([r.score for r in self.relation_results])
        all_scores.extend([r.score for r in self.yes_no_results])

        if not all_scores:
            return 0.0
        return sum(all_scores) / len(all_scores)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'overall_score': self.overall_score,
            'pointing_results': [
                {
                    'correct': r.correct,
                    'target': r.target_object,
                    'distance_error': r.distance_error,
                    'score': r.score,
                    'hit_same_category': r.hit_same_category,
                    'hit_object': r.hit_object,
                    'target_category': r.target_category
                }
                for r in self.pointing_results
            ],
            'path_results': [
                {
                    'start_correct': r.start_correct,
                    'end_correct': r.end_correct,
                    'path_valid': r.path_valid,
                    'collisions': r.collisions,
                    'efficiency': r.efficiency,
                    'score': r.score,
                    'direction': {
                        'correct': r.direction_result.direction_correct,
                        'expected': r.direction_result.expected_direction,
                        'actual': r.direction_result.actual_direction,
                        'consistency': r.direction_result.consistency_score,
                        'backtracks': r.direction_result.backtrack_count
                    } if r.direction_result else None
                }
                for r in self.path_results
            ],
            'region_results': [
                {
                    'correct': r.correct_object,
                    'target': r.target_object,
                    'iou': r.iou,
                    'score': r.score
                }
                for r in self.region_results
            ],
            'relation_results': [
                {
                    'correct': r.correct,
                    'relation': r.relation_type,
                    'entities': [r.entity_a, r.entity_b],
                    'score': r.score
                }
                for r in self.relation_results
            ],
            'yes_no_results': [
                {
                    'answer_found': r.answer_found,
                    'vlm_answer': r.vlm_answer,
                    'gt_answer': r.gt_answer,
                    'correct': r.correct,
                    'score': r.score
                }
                for r in self.yes_no_results
            ],
            'metadata': self.metadata
        }


class SpatialEvaluator:
    """
    Evaluates VLM spatial outputs against ground truth.

    Takes recorded outputs from primitive tools and compares
    against ground truth from SAM3+Depth pipeline.
    """

    def __init__(
        self,
        ground_truth: GroundTruth,
        pointing_threshold: float = 50.0,
        path_endpoint_threshold: float = 75.0,
        iou_threshold: float = 0.3
    ):
        """
        Initialize evaluator.

        Args:
            ground_truth: GroundTruth instance
            pointing_threshold: Max distance for correct pointing (pixels)
            path_endpoint_threshold: Max distance for correct path endpoints
            iou_threshold: Min IoU for correct region marking
        """
        self.gt = ground_truth
        self.pointing_threshold = pointing_threshold
        self.path_endpoint_threshold = path_endpoint_threshold
        self.iou_threshold = iou_threshold

    # ========== Yes/No Answer Evaluation ==========

    def extract_yes_no(self, response_text: str) -> Optional[bool]:
        """
        Extract yes/no answer from model response text.

        Args:
            response_text: The model's text response

        Returns:
            True for "yes", False for "no", None if couldn't parse
        """
        if not response_text:
            return None

        text_lower = response_text.lower().strip()
        import re

        # Check for explicit yes/no at the start
        if text_lower.startswith('yes'):
            return True
        if text_lower.startswith('no'):
            return False

        # Affirmative patterns (indicates YES)
        yes_patterns = [
            r'\byes\b',
            r'\bcorrect\b',
            r'\btrue\b',
            r'\baffirmative\b',
            r'\bindeed\b',  # "is indeed"
            r'\bis (in the|on the|to the)',  # "is in the background", "is on the left"
            r'\bit is\b',  # "it is on the left"
        ]

        # Negative patterns (indicates NO)
        no_patterns = [
            r'\bno\b',
            r'\bincorrect\b',
            r'\bfalse\b',
            r'\bnegative\b',
            r'\bnot\b',  # "is not", "not in the"
            r'\bisn\'t\b',
            r'\bwould not\b',
            r'\bwouldn\'t\b',
            r'rather than',  # "X rather than Y" = not Y
            r'instead of',  # "X instead of Y" = not Y
        ]

        # Check negative patterns first (they're more specific)
        for pattern in no_patterns:
            if re.search(pattern, text_lower):
                return False

        # Then check affirmative patterns
        for pattern in yes_patterns:
            if re.search(pattern, text_lower):
                return True

        return None

    def evaluate_yes_no(
        self,
        response_text: str,
        gt_answer: bool
    ) -> YesNoResult:
        """
        Evaluate a yes/no answer.

        Args:
            response_text: Model's text response
            gt_answer: Ground truth answer (True=yes, False=no)

        Returns:
            YesNoResult with evaluation
        """
        vlm_answer = self.extract_yes_no(response_text)

        return YesNoResult(
            answer_found=vlm_answer is not None,
            vlm_answer=vlm_answer,
            gt_answer=gt_answer,
            correct=vlm_answer == gt_answer if vlm_answer is not None else False
        )

    # ========== Pointing Evaluation ==========

    def _get_category(self, object_name: str) -> str:
        """Extract category from object name (e.g., 'stool_0' -> 'stool')."""
        parts = object_name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return object_name

    def _point_inside_bbox(self, point: Tuple[int, int], bbox: List[int]) -> bool:
        """Check if point is inside bounding box."""
        x1, y1, x2, y2 = bbox
        return (x1 <= point[0] <= x2) and (y1 <= point[1] <= y2)

    def evaluate_pointing(
        self,
        vlm_point: Tuple[int, int],
        target_object: str
    ) -> PointingResult:
        """
        Evaluate if VLM pointed to the correct object.

        Uses three-tier evaluation with partial credit:
        1. If point is inside target object's bbox -> correct (1.0)
        2. If point is inside any object of same category -> partial credit (0.5)
        3. Otherwise, use distance threshold for correctness

        Args:
            vlm_point: (x, y) where VLM pointed
            target_object: Object that should have been pointed to

        Returns:
            PointingResult with evaluation details including partial credit
        """
        target_category = self._get_category(target_object)
        gt_position = self.gt.get_object_position(target_object)

        if gt_position is None:
            return PointingResult(
                correct=False,
                target_object=target_object,
                vlm_point=vlm_point,
                gt_position=(0, 0),
                distance_error=float('inf'),
                threshold=self.pointing_threshold,
                target_category=target_category
            )

        # Check if point is inside target object's bounding box
        bbox = self.gt.get_object_bbox(target_object)
        inside_target_bbox = False
        if bbox:
            inside_target_bbox = self._point_inside_bbox(vlm_point, bbox)

        # Calculate distance to center
        distance = math.sqrt(
            (vlm_point[0] - gt_position[0])**2 +
            (vlm_point[1] - gt_position[1])**2
        )

        # Use relative threshold: 5% of image diagonal
        image_diagonal = math.sqrt(self.gt.image_width**2 + self.gt.image_height**2)
        relative_threshold = max(self.pointing_threshold, image_diagonal * 0.05)

        # Check if correct (exact target)
        is_correct = inside_target_bbox or (distance < relative_threshold)

        # Check for partial credit: point inside any object of same category
        hit_same_category = False
        hit_object = None

        if not is_correct:
            # Check all objects of same category
            for obj in self.gt.get_all_objects():
                if self._get_category(obj.name) == target_category and obj.name != target_object:
                    obj_bbox = self.gt.get_object_bbox(obj.name)
                    if obj_bbox and self._point_inside_bbox(vlm_point, obj_bbox):
                        hit_same_category = True
                        hit_object = obj.name
                        break

        return PointingResult(
            correct=is_correct,
            target_object=target_object,
            vlm_point=vlm_point,
            gt_position=gt_position,
            distance_error=distance,
            threshold=relative_threshold,
            hit_same_category=hit_same_category,
            hit_object=hit_object,
            target_category=target_category
        )

    def evaluate_pointing_to_relation(
        self,
        vlm_point: Tuple[int, int],
        relation: str,
        reference_object: str
    ) -> PointingResult:
        """
        Evaluate pointing for relational queries like "point to object behind X".

        Args:
            vlm_point: Where VLM pointed
            relation: Relation type (e.g., 'behind', 'left_of')
            reference_object: Reference object

        Returns:
            PointingResult
        """
        # Get objects matching the relation
        if relation == 'behind':
            valid_objects = self.gt.get_objects_behind(reference_object)
        elif relation == 'in_front_of':
            valid_objects = self.gt.get_objects_in_front_of(reference_object)
        elif relation == 'left_of':
            valid_objects = self.gt.get_objects_left_of(reference_object)
        elif relation == 'right_of':
            valid_objects = self.gt.get_objects_right_of(reference_object)
        else:
            valid_objects = []

        if not valid_objects:
            return PointingResult(
                correct=False,
                target_object=f"{relation} {reference_object}",
                vlm_point=vlm_point,
                gt_position=(0, 0),
                distance_error=float('inf'),
                threshold=self.pointing_threshold
            )

        # Check if VLM pointed near any valid object
        min_distance = float('inf')
        nearest_obj = None
        nearest_pos = (0, 0)

        for obj in valid_objects:
            dist = math.sqrt(
                (vlm_point[0] - obj.position[0])**2 +
                (vlm_point[1] - obj.position[1])**2
            )
            if dist < min_distance:
                min_distance = dist
                nearest_obj = obj
                nearest_pos = obj.position

        return PointingResult(
            correct=min_distance < self.pointing_threshold,
            target_object=nearest_obj.name if nearest_obj else "none",
            vlm_point=vlm_point,
            gt_position=nearest_pos,
            distance_error=min_distance,
            threshold=self.pointing_threshold
        )

    # ========== Path Direction Evaluation ==========

    def evaluate_path_direction(
        self,
        vlm_path: List[Tuple[int, int]],
        expected_direction: str,
        start_position: Tuple[float, float] = None
    ) -> PathDirectionResult:
        """
        Evaluate if path waypoints follow the expected direction.

        Args:
            vlm_path: List of (x, y) waypoints
            expected_direction: Expected direction ("left", "right", "forward", "backward", "nearest")
            start_position: Starting position for egocentric evaluation

        Returns:
            PathDirectionResult with direction evaluation
        """
        if not vlm_path or len(vlm_path) < 2:
            return PathDirectionResult(
                direction_correct=False,
                expected_direction=expected_direction,
                actual_direction="none",
                consistency_score=0.0,
                backtrack_count=0
            )

        # Compute overall direction from first to last waypoint
        dx_total = vlm_path[-1][0] - vlm_path[0][0]
        dy_total = vlm_path[-1][1] - vlm_path[0][1]

        # Determine actual direction
        if abs(dx_total) > abs(dy_total):
            actual_direction = "right" if dx_total > 0 else "left"
        else:
            # In image coords, y increases downward
            # "forward" = toward bottom (increasing y) = toward camera
            # "backward" = toward top (decreasing y) = away from camera
            actual_direction = "forward" if dy_total > 0 else "backward"

        # Check if direction matches expected
        direction_matches = {
            ("left", "left"): True,
            ("right", "right"): True,
            ("forward", "forward"): True,
            ("foreground", "forward"): True,
            ("backward", "backward"): True,
            ("background", "backward"): True,
            ("nearest", "forward"): True,  # nearest usually means moving forward
            ("nearest", "left"): True,     # nearest can be any direction
            ("nearest", "right"): True,
        }
        direction_correct = direction_matches.get(
            (expected_direction, actual_direction),
            expected_direction == actual_direction
        )

        # Compute consistency score (how consistently path moves in one direction)
        x_changes = []
        y_changes = []
        for i in range(len(vlm_path) - 1):
            x_changes.append(vlm_path[i+1][0] - vlm_path[i][0])
            y_changes.append(vlm_path[i+1][1] - vlm_path[i][1])

        # Count backtracks (direction reversals)
        backtrack_count = 0
        for i in range(len(x_changes) - 1):
            # Check if direction reversed on x-axis
            if x_changes[i] * x_changes[i+1] < 0:
                backtrack_count += 1
            # Check if direction reversed on y-axis
            if y_changes[i] * y_changes[i+1] < 0:
                backtrack_count += 1

        # Consistency: ratio of movement in dominant direction
        if expected_direction in ("left", "right"):
            # Check x-axis consistency
            expected_sign = -1 if expected_direction == "left" else 1
            consistent_moves = sum(1 for dx in x_changes if dx * expected_sign > 0)
            total_moves = len(x_changes)
        elif expected_direction in ("forward", "foreground", "backward", "background"):
            # Check y-axis consistency
            expected_sign = 1 if expected_direction in ("forward", "foreground") else -1
            consistent_moves = sum(1 for dy in y_changes if dy * expected_sign > 0)
            total_moves = len(y_changes)
        else:
            # For "nearest", just check no excessive backtracking
            consistent_moves = max(0, len(x_changes) - backtrack_count)
            total_moves = len(x_changes)

        consistency_score = consistent_moves / total_moves if total_moves > 0 else 0.0

        return PathDirectionResult(
            direction_correct=direction_correct,
            expected_direction=expected_direction,
            actual_direction=actual_direction,
            consistency_score=consistency_score,
            backtrack_count=backtrack_count
        )

    # ========== Path Evaluation ==========

    def evaluate_path(
        self,
        vlm_path: List[Tuple[int, int]],
        start_object: str,
        goal_object: str,
        expected_direction: str = None,
        start_position: Tuple[float, float] = None
    ) -> PathResult:
        """
        Evaluate a path drawn by VLM.

        Args:
            vlm_path: List of (x, y) waypoints
            start_object: Expected start object
            goal_object: Expected goal object

        Returns:
            PathResult with evaluation details
        """
        if not vlm_path or len(vlm_path) < 2:
            return PathResult(
                start_correct=False,
                end_correct=False,
                path_valid=False,
                collisions=[],
                efficiency=0.0,
                vlm_path=vlm_path,
                vlm_path_length=0.0,
                optimal_length=0.0
            )

        # Get ground truth positions
        start_pos = self.gt.get_object_position(start_object)
        goal_pos = self.gt.get_object_position(goal_object)

        if not start_pos or not goal_pos:
            return PathResult(
                start_correct=False,
                end_correct=False,
                path_valid=False,
                collisions=[],
                efficiency=0.0,
                vlm_path=vlm_path,
                vlm_path_length=0.0,
                optimal_length=0.0
            )

        # Check start point
        start_dist = math.sqrt(
            (vlm_path[0][0] - start_pos[0])**2 +
            (vlm_path[0][1] - start_pos[1])**2
        )
        start_correct = start_dist < self.path_endpoint_threshold

        # Check end point
        end_dist = math.sqrt(
            (vlm_path[-1][0] - goal_pos[0])**2 +
            (vlm_path[-1][1] - goal_pos[1])**2
        )
        end_correct = end_dist < self.path_endpoint_threshold

        # Check path validity (no collisions)
        path_valid, collisions = self.gt.is_path_valid(
            vlm_path,
            excluded_objects=[start_object, goal_object]
        )

        # Compute path length
        vlm_path_length = sum(
            math.sqrt(
                (vlm_path[i+1][0] - vlm_path[i][0])**2 +
                (vlm_path[i+1][1] - vlm_path[i][1])**2
            )
            for i in range(len(vlm_path) - 1)
        )

        # Optimal length (straight line)
        optimal_length = self.gt.compute_optimal_path_length(start_object, goal_object)
        optimal_length = optimal_length or vlm_path_length

        # Efficiency (optimal / actual, capped at 1.0)
        efficiency = optimal_length / vlm_path_length if vlm_path_length > 0 else 0.0
        efficiency = min(efficiency, 1.0)

        # Evaluate direction if expected_direction is provided
        direction_result = None
        if expected_direction:
            direction_result = self.evaluate_path_direction(
                vlm_path,
                expected_direction,
                start_position
            )

        return PathResult(
            start_correct=start_correct,
            end_correct=end_correct,
            path_valid=path_valid,
            collisions=collisions,
            efficiency=efficiency,
            vlm_path=vlm_path,
            vlm_path_length=vlm_path_length,
            optimal_length=optimal_length,
            direction_result=direction_result
        )

    def evaluate_egocentric_path(
        self,
        vlm_path: List[Tuple[int, int]],
        goal_object: str,
        start_position: Tuple[float, float],
        expected_direction: str
    ) -> PathResult:
        """
        Evaluate an egocentric path (starting from viewer position).

        Args:
            vlm_path: List of (x, y) waypoints
            goal_object: Expected goal object
            start_position: Viewer's starting position
            expected_direction: Expected direction ("left", "right", etc.)

        Returns:
            PathResult with evaluation details
        """
        if not vlm_path or len(vlm_path) < 2:
            return PathResult(
                start_correct=False,
                end_correct=False,
                path_valid=False,
                collisions=[],
                efficiency=0.0,
                vlm_path=vlm_path,
                vlm_path_length=0.0,
                optimal_length=0.0,
                direction_result=PathDirectionResult(
                    direction_correct=False,
                    expected_direction=expected_direction,
                    actual_direction="none",
                    consistency_score=0.0,
                    backtrack_count=0
                )
            )

        # Get ground truth positions
        goal_pos = self.gt.get_object_position(goal_object)

        if not goal_pos:
            return PathResult(
                start_correct=False,
                end_correct=False,
                path_valid=False,
                collisions=[],
                efficiency=0.0,
                vlm_path=vlm_path,
                vlm_path_length=0.0,
                optimal_length=0.0,
                direction_result=None
            )

        # Check start point (should be near viewer position)
        start_dist = math.sqrt(
            (vlm_path[0][0] - start_position[0])**2 +
            (vlm_path[0][1] - start_position[1])**2
        )
        start_correct = start_dist < self.path_endpoint_threshold

        # Check end point
        end_dist = math.sqrt(
            (vlm_path[-1][0] - goal_pos[0])**2 +
            (vlm_path[-1][1] - goal_pos[1])**2
        )
        end_correct = end_dist < self.path_endpoint_threshold

        # Check path validity (no collisions)
        path_valid, collisions = self.gt.is_path_valid(
            vlm_path,
            excluded_objects=[goal_object]
        )

        # Compute path length
        vlm_path_length = sum(
            math.sqrt(
                (vlm_path[i+1][0] - vlm_path[i][0])**2 +
                (vlm_path[i+1][1] - vlm_path[i][1])**2
            )
            for i in range(len(vlm_path) - 1)
        )

        # Optimal length (straight line from viewer to goal)
        optimal_length = math.sqrt(
            (goal_pos[0] - start_position[0])**2 +
            (goal_pos[1] - start_position[1])**2
        )
        optimal_length = optimal_length or vlm_path_length

        # Efficiency
        efficiency = optimal_length / vlm_path_length if vlm_path_length > 0 else 0.0
        efficiency = min(efficiency, 1.0)

        # Evaluate direction
        direction_result = self.evaluate_path_direction(
            vlm_path,
            expected_direction,
            start_position
        )

        return PathResult(
            start_correct=start_correct,
            end_correct=end_correct,
            path_valid=path_valid,
            collisions=collisions,
            efficiency=efficiency,
            vlm_path=vlm_path,
            vlm_path_length=vlm_path_length,
            optimal_length=optimal_length,
            direction_result=direction_result
        )

    # ========== Region Evaluation ==========

    def compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Compute Intersection over Union of two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def evaluate_region(
        self,
        vlm_bbox: List[int],
        target_object: str
    ) -> RegionResult:
        """
        Evaluate a region marked by VLM.

        Args:
            vlm_bbox: [x1, y1, x2, y2] marked by VLM
            target_object: Object that should be in the region

        Returns:
            RegionResult with evaluation details
        """
        gt_bbox = self.gt.get_object_bbox(target_object)
        gt_position = self.gt.get_object_position(target_object)

        if gt_bbox is None:
            return RegionResult(
                correct_object=False,
                target_object=target_object,
                vlm_bbox=vlm_bbox,
                gt_bbox=[0, 0, 0, 0],
                iou=0.0,
                center_distance=float('inf')
            )

        iou = self.compute_iou(vlm_bbox, gt_bbox)

        vlm_center = ((vlm_bbox[0] + vlm_bbox[2]) / 2, (vlm_bbox[1] + vlm_bbox[3]) / 2)
        center_distance = math.sqrt(
            (vlm_center[0] - gt_position[0])**2 +
            (vlm_center[1] - gt_position[1])**2
        )

        return RegionResult(
            correct_object=iou >= self.iou_threshold,
            target_object=target_object,
            vlm_bbox=vlm_bbox,
            gt_bbox=gt_bbox,
            iou=iou,
            center_distance=center_distance
        )

    # ========== Spatial Relation Evaluation ==========

    def evaluate_spatial_relation(
        self,
        vlm_answer: bool,
        relation_type: str,
        entity_a: str,
        entity_b: str
    ) -> SpatialRelationResult:
        """
        Evaluate VLM's answer to a spatial relation question.

        Args:
            vlm_answer: VLM's yes/no answer
            relation_type: Type of relation (left_of, behind, etc.)
            entity_a: First entity
            entity_b: Second entity (reference)

        Returns:
            SpatialRelationResult
        """
        # Get ground truth answer
        if relation_type == 'left_of':
            gt_answer = self.gt.is_left_of(entity_a, entity_b)
        elif relation_type == 'right_of':
            gt_answer = self.gt.is_right_of(entity_a, entity_b)
        elif relation_type == 'above':
            gt_answer = self.gt.is_above(entity_a, entity_b)
        elif relation_type == 'below':
            gt_answer = self.gt.is_below(entity_a, entity_b)
        elif relation_type == 'in_front_of':
            gt_answer = self.gt.is_in_front_of(entity_a, entity_b)
        elif relation_type == 'behind':
            gt_answer = self.gt.is_behind(entity_a, entity_b)
        elif relation_type == 'near':
            gt_answer = self.gt.is_near(entity_a, entity_b)
        else:
            gt_answer = False

        return SpatialRelationResult(
            correct=vlm_answer == gt_answer,
            relation_type=relation_type,
            entity_a=entity_a,
            entity_b=entity_b,
            vlm_answer=vlm_answer,
            gt_answer=gt_answer
        )

    # ========== Full Evaluation ==========

    def evaluate_recorded_outputs(
        self,
        recorded_outputs: RecordedOutputs,
        task_id: str,
        task_type: str,
        expected_targets: Dict[str, Any] = None
    ) -> EvaluationReport:
        """
        Evaluate all recorded outputs from a task.

        Args:
            recorded_outputs: RecordedOutputs from primitive tools
            task_id: Task identifier
            task_type: Type of task (e.g., 'navigation', 'pointing')
            expected_targets: Dict with expected targets for evaluation

        Returns:
            EvaluationReport with all results
        """
        report = EvaluationReport(
            task_id=task_id,
            task_type=task_type,
            metadata=expected_targets or {}
        )

        expected_targets = expected_targets or {}

        # Evaluate paths
        if recorded_outputs.paths:
            start_obj = expected_targets.get('start_object')
            goal_obj = expected_targets.get('goal_object')
            start_position = expected_targets.get('start_position')
            expected_direction = expected_targets.get('expected_direction')

            for path_record in recorded_outputs.paths:
                path_points = path_record['points']

                if task_type == 'egocentric_path_planning' and start_position and goal_obj:
                    # Egocentric: starts from viewer position
                    result = self.evaluate_egocentric_path(
                        path_points,
                        goal_obj,
                        tuple(start_position),
                        expected_direction or 'nearest'
                    )
                    report.path_results.append(result)
                elif start_obj and goal_obj:
                    # Allocentric: object to object
                    result = self.evaluate_path(
                        path_points,
                        start_obj,
                        goal_obj,
                        expected_direction=expected_direction,
                        start_position=tuple(start_position) if start_position else None
                    )
                    report.path_results.append(result)

        # Evaluate points
        if recorded_outputs.points:
            target_obj = expected_targets.get('target_object')
            valid_objects = expected_targets.get('valid_objects', [])
            relation = expected_targets.get('relation') or expected_targets.get('relation_type')
            reference = expected_targets.get('reference_object')

            for point_record in recorded_outputs.points:
                vlm_point = (point_record['x'], point_record['y'])

                if valid_objects:
                    # For egocentric/allocentric QA: check if pointing to any valid object
                    best_result = None
                    min_distance = float('inf')

                    for valid_obj in valid_objects:
                        result = self.evaluate_pointing(vlm_point, valid_obj)
                        if result.distance_error < min_distance:
                            min_distance = result.distance_error
                            best_result = result

                    if best_result:
                        report.pointing_results.append(best_result)
                elif relation and reference:
                    result = self.evaluate_pointing_to_relation(
                        vlm_point, relation, reference
                    )
                    report.pointing_results.append(result)
                elif target_obj:
                    result = self.evaluate_pointing(vlm_point, target_obj)
                    report.pointing_results.append(result)
                else:
                    # Use label as target if available
                    label = point_record.get('label', '')
                    result = self.evaluate_pointing(vlm_point, label)
                    report.pointing_results.append(result)

        # Evaluate regions
        if recorded_outputs.regions:
            for region_record in recorded_outputs.regions:
                target = region_record.get('label', expected_targets.get('target_object', ''))
                result = self.evaluate_region(region_record['bbox'], target)
                report.region_results.append(result)

        return report

    def evaluate_with_response(
        self,
        recorded_outputs: RecordedOutputs,
        response_text: str,
        task_id: str,
        task_type: str,
        expected_targets: Dict[str, Any] = None
    ) -> EvaluationReport:
        """
        Evaluate outputs including yes/no answer from response text.

        Args:
            recorded_outputs: RecordedOutputs from primitive tools
            response_text: Model's text response
            task_id: Task identifier
            task_type: Type of task
            expected_targets: Dict with expected targets

        Returns:
            EvaluationReport with all results including yes/no
        """
        # First do standard evaluation
        report = self.evaluate_recorded_outputs(
            recorded_outputs, task_id, task_type, expected_targets
        )

        expected_targets = expected_targets or {}

        # Evaluate yes/no if required
        if expected_targets.get('requires_yes_no') and 'ground_truth_answer' in expected_targets:
            gt_answer = expected_targets['ground_truth_answer']
            yes_no_result = self.evaluate_yes_no(response_text, gt_answer)
            report.yes_no_results.append(yes_no_result)

        return report
