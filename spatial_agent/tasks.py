#!/usr/bin/env python3
"""
tasks.py - Spatial intelligence task definitions

Defines task types for evaluating VLM spatial intelligence:

Task Types:
- Object Localization: Point to a described object
- Egocentric Path Planning: Navigate using viewer-relative directions
- Allocentric Path Planning: Navigate between objects
- Egocentric Spatial QA: Spatial questions from viewer's perspective
- Allocentric Spatial QA: Spatial relations from a viewpoint perspective

Each task can include an annotated image with visual markers:
- Viewpoint marker (position + facing arrow)
- Start/end markers for path planning
- Object highlights for spatial QA
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from PIL import Image

from .ground_truth import GroundTruth, ObjectInfo
from .object_descriptor import ObjectDescriptorGenerator
from .viewpoint_annotator import (
    ViewpointAnnotator,
    annotate_for_egocentric_task,
    annotate_for_allocentric_path,
    annotate_for_allocentric_qa,
    get_standard_viewpoints,
    compute_facing_toward_object
)


class TaskType(Enum):
    """Types of spatial intelligence tasks (4 total)."""
    OBJECT_LOCALIZATION = "object_localization"      # Point to specified object
    PATH_PLANNING = "path_planning"                   # Draw path from START HERE to target
    EGOCENTRIC_SPATIAL_QA = "egocentric_spatial_qa"  # Spatial questions from camera view
    ALLOCENTRIC_SPATIAL_QA = "allocentric_spatial_qa" # Spatial questions from marked viewpoint

    # Legacy aliases for backward compatibility
    EGOCENTRIC_PATH_PLANNING = "path_planning"
    ALLOCENTRIC_PATH_PLANNING = "path_planning"


class Difficulty(Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class SpatialTask:
    """A spatial intelligence task with optional annotated image."""
    task_id: str
    task_type: TaskType
    prompt: str
    instruction: str
    expected_tool: str  # Primary tool expected to be used
    expected_targets: Dict[str, Any]
    difficulty: Difficulty
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional annotated image (PIL Image or path)
    annotated_image: Optional[Union[Image.Image, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'prompt': self.prompt,
            'instruction': self.instruction,
            'expected_tool': self.expected_tool,
            'expected_targets': self.expected_targets,
            'difficulty': self.difficulty.value,
            'metadata': self.metadata
        }


class TaskGenerator:
    """
    Generates spatial intelligence tasks from ground truth.

    Uses ObjectDescriptorGenerator for natural language object references
    instead of internal IDs like cabinet_0, stool_1.

    Can generate annotated images with visual markers for:
    - Viewpoint position and facing direction
    - Start/end markers for path planning
    - Object highlights for spatial QA
    """

    # Egocentric directions
    EGOCENTRIC_DIRECTIONS = ['left', 'right', 'foreground', 'background', 'nearest']

    # Allocentric relations (full scale)
    ALLOCENTRIC_RELATIONS = ['left', 'right', 'in_front', 'behind', 'above', 'below', 'near', 'far']

    # Facing directions for allocentric reasoning
    FACING_DIRECTIONS = ['camera', 'away', 'left', 'right']

    def __init__(
        self,
        ground_truth: GroundTruth,
        image_path: Optional[str] = None,
        seed: Optional[int] = None,
        annotate_images: bool = True
    ):
        """
        Initialize task generator.

        Args:
            ground_truth: GroundTruth instance
            image_path: Path to the scene image (for annotation)
            seed: Random seed for reproducibility
            annotate_images: Whether to generate annotated images for tasks
        """
        self.gt = ground_truth
        self.all_objects = ground_truth.get_all_objects()
        self.descriptor_gen = ObjectDescriptorGenerator(ground_truth)
        self.task_counter = 0

        # Image path for annotation
        self.image_path = image_path
        self.annotate_images = annotate_images and (image_path is not None)

        # Only use objects with natural descriptors
        describable_names = set(self.descriptor_gen.get_describable_objects())
        self.objects = [obj for obj in self.all_objects if obj.name in describable_names]

        if not self.objects:
            # Fallback to all objects if none are describable
            self.objects = self.all_objects

        # Viewer position (bottom-middle for egocentric tasks)
        # This represents where the viewer is "standing" in the scene
        self.viewer_position = (
            ground_truth.image_width / 2,
            ground_truth.image_height * 0.85  # Bottom of image = viewer's position
        )

        # Standard viewpoint positions for variety
        self.viewpoint_positions = get_standard_viewpoints(
            (ground_truth.image_width, ground_truth.image_height)
        )

        if seed is not None:
            random.seed(seed)

    def _next_task_id(self) -> str:
        self.task_counter += 1
        return f"task_{self.task_counter:04d}"

    def _get_descriptor(self, object_name: str) -> str:
        """Get natural language descriptor for an object."""
        return self.descriptor_gen.get_descriptor_string(object_name)

    def _get_category(self, object_name: str) -> str:
        """Get base category from object name (cabinet_0 -> cabinet)."""
        return self.descriptor_gen._get_base_category(object_name)

    def _get_unique_objects(self) -> List[ObjectInfo]:
        """Get objects that are unique in their category (only one of that type)."""
        unique = []
        for obj in self.objects:
            category = self._get_category(obj.name)
            # Check if this is the only object in its category
            if self.descriptor_gen.by_category[category] == [obj]:
                unique.append(obj)
            else:
                # Also include if descriptor marks it as unique
                desc = self.descriptor_gen.get_descriptor(obj.name)
                if desc and desc.is_unique:
                    unique.append(obj)
        return unique

    def _get_objects_by_category(self) -> Dict[str, List[ObjectInfo]]:
        """Get objects grouped by category."""
        return dict(self.descriptor_gen.by_category)

    def _get_obstacle_description(self, exclude_objects: List[str] = None) -> str:
        """
        Generate a description of obstacles in the scene for path planning.

        Args:
            exclude_objects: Object names to exclude from obstacle list (e.g., start/goal)

        Returns:
            Human-readable description of obstacles and their locations
        """
        exclude_objects = exclude_objects or []
        obstacles = []

        for obj in self.all_objects:
            if obj.name in exclude_objects:
                continue

            category = self._get_category(obj.name)
            x, y = obj.position

            # Describe position in scene quadrants
            img_w = self.gt.image_width
            img_h = self.gt.image_height

            h_pos = "left" if x < img_w * 0.4 else "right" if x > img_w * 0.6 else "center"
            v_pos = "top" if y < img_h * 0.4 else "bottom" if y > img_h * 0.6 else "middle"

            obstacles.append(f"- {category} ({h_pos}, {v_pos})")

        if not obstacles:
            return "No major obstacles detected."

        return "Known obstacles to avoid:\n" + "\n".join(obstacles)

    # ==========================================================================
    # OBJECT LOCALIZATION
    # ==========================================================================

    def generate_localization_task(
        self,
        target_object: Optional[str] = None
    ) -> SpatialTask:
        """
        Generate task: "Point to the [described object]"

        Uses natural language descriptors like "the leftmost cabinet"
        instead of internal IDs like "cabinet_1".

        No annotation needed - just clear instructions.
        """
        if target_object is None:
            obj = random.choice(self.objects)
            target_object = obj.name

        descriptor = self._get_descriptor(target_object)

        # Simplified prompt - let model understand spatially
        prompt = f"""You are currently viewing this scene.

TASK: Point to {descriptor}.

Use point_to to indicate its location."""

        instruction = f"Point to {descriptor}"

        return SpatialTask(
            task_id=self._next_task_id(),
            task_type=TaskType.OBJECT_LOCALIZATION,
            prompt=prompt,
            instruction=instruction,
            expected_tool="point_to",
            expected_targets={
                'target_object': target_object,
                'target_descriptor': descriptor
            },
            difficulty=Difficulty.EASY
        )

    # ==========================================================================
    # PATH PLANNING (unified - always uses START HERE marker)
    # ==========================================================================

    def generate_path_task(
        self,
        target_object: Optional[str] = None,
        start_position: Optional[Tuple[float, float]] = None
    ) -> SpatialTask:
        """
        Generate path planning task.

        Shows "START HERE" marker at the starting position.
        Model must draw a path from START HERE to the target object.

        Args:
            target_object: Target object name (random if None)
            start_position: Starting position (default: bottom-center of image)
        """
        # Select target object
        if target_object is None:
            target_obj = random.choice(self.objects)
        else:
            target_obj = self.gt.resolve_object(target_object)
            if target_obj is None:
                target_obj = random.choice(self.objects)

        target_descriptor = self._get_descriptor(target_obj.name)

        # Use provided start position or default viewer position
        start_pos = start_position if start_position else self.viewer_position

        # Compute expected direction for evaluation (not shown to model)
        target_pos = self.gt.get_object_position(target_obj.name)
        dx = target_pos[0] - start_pos[0]
        dy = target_pos[1] - start_pos[1]

        if abs(dx) > abs(dy):
            expected_direction = "right" if dx > 0 else "left"
        else:
            expected_direction = "backward" if dy < 0 else "forward"

        # Create annotated image with START HERE marker
        annotated_image = None
        if self.annotate_images:
            annotator = ViewpointAnnotator(self.image_path)
            annotator.add_start_marker(start_pos, label="START HERE")
            annotated_image = annotator.get_image()

        # Clear prompt referencing the visual marker
        prompt = f"""Look at the image. Your starting position is marked "START HERE".

TASK: Draw a walking path from "START HERE" to {target_descriptor}. Avoid any obstacles in the way.

Use the draw_path tool with a list of [x, y] waypoint coordinates."""

        instruction = f"Path: START HERE → {target_descriptor}"

        return SpatialTask(
            task_id=self._next_task_id(),
            task_type=TaskType.PATH_PLANNING,
            prompt=prompt,
            instruction=instruction,
            expected_tool="draw_path",
            expected_targets={
                'goal_object': target_obj.name,
                'goal_descriptor': target_descriptor,
                'expected_direction': expected_direction,
                'start_position': start_pos
            },
            difficulty=Difficulty.MEDIUM,
            metadata={
                'start_position': start_pos,
                'target_position': target_pos
            },
            annotated_image=annotated_image
        )

    # Alias for backward compatibility
    def generate_egocentric_path_task(self, target_object: Optional[str] = None) -> SpatialTask:
        """Alias for generate_path_task (backward compatibility)."""
        return self.generate_path_task(target_object=target_object)

    # ==========================================================================
    # ALLOCENTRIC PATH PLANNING
    # ==========================================================================

    def generate_allocentric_path_task(
        self,
        viewpoint_position: Optional[Tuple[float, float]] = None,
        viewpoint_name: Optional[str] = None,
        facing: Optional[str] = None,
        start_object: Optional[str] = None,
        goal_object: Optional[str] = None
    ) -> SpatialTask:
        """
        Generate allocentric path planning task with viewpoint marker.

        ALLOCENTRIC = Hypothetical viewpoint (different from camera).
        Image is annotated with viewpoint marker showing position + facing.

        Task: "You are at the marked position. Draw a path from A to B."
        - Model must understand the scene from the viewpoint
        - Model must plan a valid walking path between objects
        - NO direction hints given

        Similar structure to Allocentric Spatial QA but for path planning.
        """
        if start_object is None or goal_object is None:
            if len(self.objects) < 2:
                raise ValueError("Need at least 2 objects for path task")
            objs = random.sample(self.objects, 2)
            start_object = objs[0].name
            goal_object = objs[1].name

        start_descriptor = self._get_descriptor(start_object)
        goal_descriptor = self._get_descriptor(goal_object)

        # Get object positions
        start_pos = self.gt.get_object_position(start_object)
        goal_pos = self.gt.get_object_position(goal_object)

        # Select viewpoint position (different from camera)
        if viewpoint_position is None:
            viewpoint_names = ['center_bottom', 'bottom_left', 'bottom_right']
            viewpoint_name = random.choice(viewpoint_names)
            viewpoint_position = self.viewpoint_positions[viewpoint_name]
        elif viewpoint_name is None:
            viewpoint_name = "marked position"

        # Select facing direction
        if facing is None:
            facing = random.choice(self.FACING_DIRECTIONS)

        # Compute expected direction from start to goal (for evaluation only)
        dx = goal_pos[0] - start_pos[0]
        expected_direction = "right" if dx > 0 else "left"

        # Create annotated image with viewpoint marker
        annotated_image = None
        if self.annotate_images:
            annotated_image = annotate_for_allocentric_qa(
                self.image_path,
                viewpoint_position=viewpoint_position,
                facing=facing
            )

        # Category names for cleaner reference
        start_category = self._get_category(start_object)
        goal_category = self._get_category(goal_object)

        # Simplified prompt - use arrow orientation, no obstacle list
        prompt = f"""Imagine you are standing at the marked position (X) with the arrow orientation.

TASK: Draw a walking path from {start_descriptor} to {goal_descriptor}. Make sure to avoid obstacles.

Use draw_path with waypoint coordinates to show the route."""

        instruction = f"From viewpoint: Draw path from {start_category} to {goal_category}"

        return SpatialTask(
            task_id=self._next_task_id(),
            task_type=TaskType.ALLOCENTRIC_PATH_PLANNING,
            prompt=prompt,
            instruction=instruction,
            expected_tool="draw_path",
            expected_targets={
                'viewpoint_position': viewpoint_position,
                'viewpoint_name': viewpoint_name,
                'facing': facing,
                'start_object': start_object,
                'goal_object': goal_object,
                'start_descriptor': start_descriptor,
                'goal_descriptor': goal_descriptor,
                'expected_direction': expected_direction
            },
            difficulty=Difficulty.HARD,
            metadata={
                'viewpoint_position': viewpoint_position,
                'facing': facing,
                'start_category': start_category,
                'goal_category': goal_category
            },
            annotated_image=annotated_image
        )

    # ==========================================================================
    # EGOCENTRIC SPATIAL QA
    # ==========================================================================

    def generate_egocentric_qa_task(
        self,
        direction: Optional[str] = None,
        target_object: Optional[str] = None
    ) -> SpatialTask:
        """
        Generate egocentric spatial QA task.

        EGOCENTRIC = Camera viewpoint (what the model directly sees).
        No viewpoint marker needed - the image itself IS the viewpoint.

        Questions from viewer's perspective:
        - "Is the table on your left?" -> yes/no + point to the table
        - "Is the person in the foreground?" -> yes/no + point to the person

        The direction IS the question being tested (not a hint).

        Always requires:
        1. A yes/no answer
        2. Pointing to the object
        """
        if direction is None:
            direction = random.choice(self.EGOCENTRIC_DIRECTIONS)

        # Get unique objects (only one of their category) to avoid position hints
        unique_objects = self._get_unique_objects()

        if not unique_objects:
            # Fallback: use any object but this may give hints
            unique_objects = self.objects

        # Pick a specific object to ask about
        if target_object is None:
            obj = random.choice(unique_objects)
            target_object = obj.name

        # Use category name only (no position descriptors)
        category = self._get_category(target_object)
        object_reference = f"the {category}"
        direction_phrase = self._direction_to_phrase(direction)

        # Get objects in that direction
        objects_in_dir = self.descriptor_gen.get_objects_in_direction(
            direction, self.viewer_position
        )

        # Check if the target object is actually in that direction
        target_in_direction = any(o.name == target_object for o in objects_in_dir)
        ground_truth_answer = target_in_direction

        # Simplified prompt - no direction reference table
        prompt = f"""Look at the image from your current viewing angle (as if you are the camera).

QUESTION: Is {object_reference} {direction_phrase}?

Answer YES or NO, then use point_to to indicate where {object_reference} is located."""

        instruction = f"Egocentric QA: Is {object_reference} {direction_phrase}?"

        return SpatialTask(
            task_id=self._next_task_id(),
            task_type=TaskType.EGOCENTRIC_SPATIAL_QA,
            prompt=prompt,
            instruction=instruction,
            expected_tool="point_to",
            expected_targets={
                'direction': direction,
                'target_object': target_object,
                'object_reference': object_reference,
                'ground_truth_answer': ground_truth_answer,
                'requires_yes_no': True
            },
            difficulty=Difficulty.MEDIUM,
            metadata={
                'question_type': 'is_object_in_direction',
                'direction': direction,
                'category': category
            }
            # No annotated_image - camera view IS the egocentric viewpoint
        )

    def _direction_to_phrase(self, direction: str) -> str:
        """Convert direction to natural phrase."""
        phrases = {
            'left': 'on your left',
            'right': 'on your right',
            'foreground': 'in the foreground',
            'front': 'in the foreground',
            'background': 'in the background',
            'back': 'in the background',
            'nearest': 'nearest to you',
            'farthest': 'farthest from you'
        }
        return phrases.get(direction, direction)

    # ==========================================================================
    # ALLOCENTRIC SPATIAL QA (True Perspective-Taking)
    # ==========================================================================

    def generate_allocentric_qa_task(
        self,
        facing: Optional[str] = None,
        reference_object: Optional[str] = None,
        target_object: Optional[str] = None,
        relation: Optional[str] = None
    ) -> SpatialTask:
        """
        Generate allocentric spatial QA task using reference object as anchor.

        ALLOCENTRIC REASONING (Reference Object as Anchor):
        1. REFERENCE OBJECT: The anchor point AND viewpoint
        2. FACING: Direction the reference object is facing
        3. TARGET: The object we're querying about
        4. RELATION: left/right/in_front/behind from reference's perspective

        Example:
        "The person is facing the camera. From the person's perspective,
         is the lamp to their left?"

        The reference object's position becomes the viewpoint for computing
        spatial relations. This is more intuitive and robust.

        Always requires:
        1. Pointing to both reference and target objects
        2. A yes/no answer about the spatial relation
        """
        # Get objects for task generation
        available_objects = self.objects if len(self.objects) >= 2 else self.all_objects

        if len(available_objects) < 2:
            raise ValueError("Need at least 2 objects for allocentric QA")

        # Group by category for diverse selection
        by_category: Dict[str, List] = {}
        for obj in available_objects:
            cat = self._get_category(obj.name)
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(obj)

        categories = list(by_category.keys())
        if len(categories) < 2:
            raise ValueError("Need at least 2 different object categories")

        # Select reference and target from different categories
        if reference_object is None or target_object is None:
            selected_cats = random.sample(categories, 2)
            reference_object = random.choice(by_category[selected_cats[0]]).name
            target_object = random.choice(by_category[selected_cats[1]]).name

        ref_category = self._get_category(reference_object)
        target_category = self._get_category(target_object)

        # Get object positions
        ref_pos = self.gt.get_object_position(reference_object)
        target_pos = self.gt.get_object_position(target_object)

        # Reference object position IS the viewpoint
        viewpoint_position = ref_pos

        # Select facing direction for the reference object
        if facing is None:
            facing = random.choice(self.FACING_DIRECTIONS)

        # Select relation to query
        if relation is None:
            relation = random.choice(['left', 'right', 'in_front', 'behind'])

        # Compute ground truth from reference object's perspective
        gt_answer = self._compute_allocentric_relation(
            viewpoint_position, facing, ref_pos, target_pos, relation
        )

        # Create annotated image with facing arrow on reference object
        annotated_image = None
        if self.annotate_images:
            annotated_image = annotate_for_allocentric_qa(
                self.image_path,
                viewpoint_position=viewpoint_position,
                facing=facing
            )

        # Build references
        ref_reference = f"the {ref_category}"
        target_reference = f"the {target_category}"

        # Format relation for question (from reference's perspective)
        if relation == "left":
            relation_phrase = "to their left"
        elif relation == "right":
            relation_phrase = "to their right"
        elif relation == "in_front":
            relation_phrase = "in front of them"
        elif relation == "behind":
            relation_phrase = "behind them"
        else:
            relation_phrase = relation

        # Clear prompt using reference object as anchor with indicated direction
        prompt = f"""Look at the image. Notice the arrow on {ref_reference} showing its facing direction.

TASK:
1. Point to {ref_reference} (the anchor object with the arrow)
2. Point to {target_reference} (the target object)

QUESTION: From {ref_reference}'s perspective, with the indicated direction, is {target_reference} {relation_phrase}?
Answer YES or NO after pointing to both objects."""

        instruction = f"Allocentric QA: From {ref_category}'s perspective (with indicated direction), is {target_category} {relation_phrase}?"

        return SpatialTask(
            task_id=self._next_task_id(),
            task_type=TaskType.ALLOCENTRIC_SPATIAL_QA,
            prompt=prompt,
            instruction=instruction,
            expected_tool="point_to",  # Expected to be called twice
            expected_targets={
                'viewpoint_position': viewpoint_position,  # Same as reference position
                'facing': facing,
                'reference_object': reference_object,
                'target_object': target_object,
                'relation': relation,
                'ground_truth_answer': gt_answer,
                'requires_yes_no': True,
                'requires_dual_point': True  # Both objects must be pointed to
            },
            difficulty=Difficulty.HARD,
            metadata={
                'question_type': 'reference_anchored_allocentric',
                'facing': facing,
                'relation': relation,
                'ref_category': ref_category,
                'target_category': target_category
            },
            annotated_image=annotated_image
        )

    def _compute_allocentric_relation(
        self,
        viewpoint: Tuple[float, float],
        facing: str,
        reference_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        relation: str
    ) -> bool:
        """
        Compute spatial relation in viewpoint's reference frame.

        Transforms positions based on viewpoint location and facing direction,
        then evaluates the spatial relation.

        Args:
            viewpoint: (x, y) viewpoint position
            facing: "camera", "away", "left", "right"
            reference_pos: (x, y) reference object position
            target_pos: (x, y) target object position
            relation: "left", "right", "in_front", "behind"

        Returns:
            True if the relation holds from the viewpoint's perspective
        """
        # Transform to viewpoint-centered coordinates
        ref_x = reference_pos[0] - viewpoint[0]
        ref_y = reference_pos[1] - viewpoint[1]
        target_x = target_pos[0] - viewpoint[0]
        target_y = target_pos[1] - viewpoint[1]

        # Rotate based on facing direction
        # Standard: facing "camera" means facing up (negative y in image coords)
        if facing == "camera":
            # No rotation needed - standard image coordinates
            # Left = negative x, Right = positive x
            pass
        elif facing == "away":
            # 180° rotation: flip both x and y
            ref_x, ref_y = -ref_x, -ref_y
            target_x, target_y = -target_x, -target_y
        elif facing == "left":
            # 90° CCW: (x, y) -> (y, -x)
            ref_x, ref_y = ref_y, -ref_x
            target_x, target_y = target_y, -target_x
        elif facing == "right":
            # 90° CW: (x, y) -> (-y, x)
            ref_x, ref_y = -ref_y, ref_x
            target_x, target_y = -target_y, target_x

        # Threshold for comparison (5% of image dimension)
        threshold = max(self.gt.image_width, self.gt.image_height) * 0.05

        # Evaluate relation in transformed coordinates
        if relation == "left":
            # Target is to the left of reference
            return target_x < ref_x - threshold
        elif relation == "right":
            # Target is to the right of reference
            return target_x > ref_x + threshold
        elif relation == "in_front":
            # Target is in front of (closer to viewpoint than) reference
            # In transformed coords, closer to viewpoint = smaller distance from origin
            target_dist = math.sqrt(target_x**2 + target_y**2)
            ref_dist = math.sqrt(ref_x**2 + ref_y**2)
            return target_dist < ref_dist - threshold
        elif relation == "behind":
            # Target is behind (farther from viewpoint than) reference
            target_dist = math.sqrt(target_x**2 + target_y**2)
            ref_dist = math.sqrt(ref_x**2 + ref_y**2)
            return target_dist > ref_dist + threshold

        return False

    def _check_relation(self, a: str, b: str, relation: str) -> bool:
        """Check if relation holds between objects."""
        if relation == 'left_of':
            return self.gt.is_left_of(a, b)
        elif relation == 'right_of':
            return self.gt.is_right_of(a, b)
        elif relation == 'above':
            return self.gt.is_above(a, b)
        elif relation == 'below':
            return self.gt.is_below(a, b)
        elif relation == 'behind':
            return self.gt.is_behind(a, b)
        elif relation == 'in_front_of':
            return self.gt.is_in_front_of(a, b)
        return False

    def _check_perspective_relation(
        self,
        reference_entity: str,
        target_entity: str,
        direction: str
    ) -> bool:
        """
        Check if target is in the given direction from reference's perspective.

        For simplicity, we assume the reference entity is "looking at" the scene
        from their position. This means:
        - "left": target's x < reference's x (target is to the left in the image)
        - "right": target's x > reference's x (target is to the right in the image)
        - "in front": target has lower z-order (closer to camera from reference)
        - "behind": target has higher z-order (farther from camera from reference)

        Args:
            reference_entity: The entity providing the viewpoint
            target_entity: The entity we're asking about
            direction: "left", "right", "in front", or "behind"

        Returns:
            True if target is in the specified direction from reference
        """
        ref_pos = self.gt.get_object_position(reference_entity)
        target_pos = self.gt.get_object_position(target_entity)

        if ref_pos is None or target_pos is None:
            return False

        ref_obj = self.gt.resolve_object(reference_entity)
        target_obj = self.gt.resolve_object(target_entity)

        if ref_obj is None or target_obj is None:
            return False

        # Horizontal threshold for left/right (5% of image width)
        h_threshold = self.gt.image_width * 0.05

        if direction == 'left':
            # Target is to the left of reference (smaller x)
            return target_pos[0] < ref_pos[0] - h_threshold
        elif direction == 'right':
            # Target is to the right of reference (larger x)
            return target_pos[0] > ref_pos[0] + h_threshold
        elif direction == 'in front':
            # Target is closer to camera (HIGHER z-order in this dataset)
            # Note: z-order appears inverted - higher z = closer to camera
            return target_obj.z_order > ref_obj.z_order
        elif direction == 'behind':
            # Target is farther from camera (LOWER z-order in this dataset)
            return target_obj.z_order < ref_obj.z_order

        return False

    def _find_objects_with_relation(self, reference: str, relation: str) -> List[str]:
        """Find all objects that have the given relation to reference."""
        valid = []
        for obj in self.objects:
            if obj.name == reference:
                continue
            if self._check_relation(obj.name, reference, relation):
                valid.append(obj.name)
        return valid

    # ==========================================================================
    # BATCH GENERATION
    # ==========================================================================

    def generate_task_suite(
        self,
        localization_count: int = 3,
        egocentric_path_count: int = 3,
        allocentric_path_count: int = 3,
        egocentric_qa_count: int = 3,
        allocentric_qa_count: int = 3
    ) -> Dict[str, List[SpatialTask]]:
        """
        Generate a full suite of tasks.

        Returns:
            Dict mapping task type to list of tasks
        """
        tasks = {
            'localization': [],
            'egocentric_path': [],
            'allocentric_path': [],
            'egocentric_qa': [],
            'allocentric_qa': []
        }

        # Localization - use varied objects
        sampled_objects = random.sample(
            self.objects,
            min(localization_count, len(self.objects))
        )
        for obj in sampled_objects:
            tasks['localization'].append(
                self.generate_localization_task(obj.name)
            )

        # Egocentric Path Planning - select different target objects
        # No direction parameter - model must locate objects visually
        sampled_targets = random.sample(
            self.objects,
            min(egocentric_path_count, len(self.objects))
        )
        for obj in sampled_targets:
            tasks['egocentric_path'].append(
                self.generate_egocentric_path_task(target_object=obj.name)
            )

        # Allocentric Path Planning
        for _ in range(allocentric_path_count):
            tasks['allocentric_path'].append(
                self.generate_allocentric_path_task()
            )

        # Egocentric QA - vary directions
        directions_used = []
        for _ in range(egocentric_qa_count):
            available = [d for d in self.EGOCENTRIC_DIRECTIONS if d not in directions_used]
            if not available:
                available = self.EGOCENTRIC_DIRECTIONS
            direction = random.choice(available)
            directions_used.append(direction)
            tasks['egocentric_qa'].append(
                self.generate_egocentric_qa_task(direction=direction)
            )

        # Allocentric QA (perspective-taking) - vary facing directions and relations
        facings_used = []
        relations = ['left', 'right', 'in_front', 'behind']
        for _ in range(allocentric_qa_count):
            # Vary facing directions
            available = [f for f in self.FACING_DIRECTIONS if f not in facings_used]
            if not available:
                available = self.FACING_DIRECTIONS
            facing = random.choice(available)
            facings_used.append(facing)
            relation = random.choice(relations)
            tasks['allocentric_qa'].append(
                self.generate_allocentric_qa_task(facing=facing, relation=relation)
            )

        return tasks

    def generate_mixed_suite(self, total_tasks: int = 20) -> List[SpatialTask]:
        """
        Generate a mixed suite of tasks across 4 task types.

        Task types:
        1. Object Localization - Point to objects
        2. Path Planning - Draw path from START HERE to target
        3. Egocentric Spatial QA - Spatial questions from camera view
        4. Allocentric Spatial QA - Spatial questions from marked viewpoint

        Distributes tasks roughly equally across the 4 types.
        """
        per_type = max(1, total_tasks // 4)
        remainder = total_tasks - (per_type * 4)

        tasks = []

        # Object Localization
        loc_count = per_type + (1 if remainder > 0 else 0)
        for _ in range(loc_count):
            try:
                tasks.append(self.generate_localization_task())
            except:
                pass

        # Path Planning
        path_count = per_type + (1 if remainder > 1 else 0)
        for _ in range(path_count):
            try:
                tasks.append(self.generate_path_task())
            except:
                pass

        # Egocentric Spatial QA
        ego_qa_count = per_type + (1 if remainder > 2 else 0)
        for _ in range(ego_qa_count):
            try:
                tasks.append(self.generate_egocentric_qa_task())
            except:
                pass

        # Allocentric Spatial QA
        allo_qa_count = per_type + (1 if remainder > 3 else 0)
        for _ in range(allo_qa_count):
            try:
                tasks.append(self.generate_allocentric_qa_task())
            except:
                pass

        random.shuffle(tasks)
        return tasks[:total_tasks]

    def generate_all_tasks_flat(self, **kwargs) -> List[SpatialTask]:
        """Generate all tasks as a flat list."""
        suite = self.generate_task_suite(**kwargs)
        all_tasks = []
        for task_list in suite.values():
            all_tasks.extend(task_list)
        return all_tasks
