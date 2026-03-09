#!/usr/bin/env python3
"""
ground_truth.py - Ground truth from SAM3 + Depth pipeline

Provides ground truth spatial information for evaluating VLM outputs:
- Object positions (from SAM3 bboxes)
- Object depth ordering (from depth model)
- Spatial relations (computed from positions/depth)
- Path validity (obstacle avoidance)
"""

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


@dataclass
class ObjectInfo:
    """Ground truth information for a single object."""
    id: str
    name: str
    category: str
    position: Tuple[float, float]  # bbox center
    bbox: List[int]  # [x1, y1, x2, y2]
    z_order: int
    relative_depth: float

    @property
    def center(self) -> Tuple[float, float]:
        return self.position

    @property
    def bbox_area(self) -> int:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


class GroundTruth:
    """
    Ground truth derived from SAM3 + Depth pipeline.

    Provides:
    - Object positions and bounding boxes
    - Spatial relations between objects
    - Path validity checking
    """

    # Thresholds for spatial relation computation
    HORIZONTAL_THRESHOLD_RATIO = 0.05  # 5% of image width
    VERTICAL_THRESHOLD_RATIO = 0.05    # 5% of image height
    PROXIMITY_NEAR_RATIO = 0.15        # 15% of image diagonal
    PROXIMITY_FAR_RATIO = 0.40         # 40% of image diagonal

    def __init__(self, spatial_graph_path: str):
        """
        Initialize ground truth from spatial_graph.json.

        Args:
            spatial_graph_path: Path to spatial_graph.json from SAM3+Depth pipeline
        """
        self.graph_path = Path(spatial_graph_path)

        with open(spatial_graph_path, 'r') as f:
            self.graph_data = json.load(f)

        # Image dimensions
        self.image_size = tuple(self.graph_data.get('image_size', [1000, 1000]))
        self.image_height = self.image_size[0]
        self.image_width = self.image_size[1]
        self.image_diagonal = math.sqrt(self.image_width**2 + self.image_height**2)

        # Compute thresholds
        self.horizontal_threshold = self.HORIZONTAL_THRESHOLD_RATIO * self.image_width
        self.vertical_threshold = self.VERTICAL_THRESHOLD_RATIO * self.image_height
        self.proximity_near = self.PROXIMITY_NEAR_RATIO * self.image_diagonal
        self.proximity_far = self.PROXIMITY_FAR_RATIO * self.image_diagonal

        # Build object dictionary
        self.objects: Dict[str, ObjectInfo] = {}
        self._name_to_id: Dict[str, str] = {}

        for node in self.graph_data.get('nodes', []):
            obj = ObjectInfo(
                id=node['id'],
                name=node.get('name', node['id']),
                category=node.get('category', 'unknown'),
                position=tuple(node['bbox_center']),
                bbox=node.get('bbox', [0, 0, 0, 0]),
                z_order=node['z_order'],
                relative_depth=node.get('relative_depth', 0.0)
            )
            self.objects[obj.id] = obj
            self._name_to_id[obj.name] = obj.id
            # Also index by category for convenience
            if obj.category not in self._name_to_id:
                self._name_to_id[obj.category] = obj.id

    def resolve_object(self, identifier: str) -> Optional[ObjectInfo]:
        """
        Resolve object by ID or name.

        Args:
            identifier: Object ID (e.g., 'entity_0') or name (e.g., 'table_0')

        Returns:
            ObjectInfo or None if not found
        """
        # Try direct ID lookup
        if identifier in self.objects:
            return self.objects[identifier]

        # Try name lookup
        if identifier in self._name_to_id:
            return self.objects[self._name_to_id[identifier]]

        # Fuzzy match
        identifier_lower = identifier.lower()
        for name, obj_id in self._name_to_id.items():
            if identifier_lower in name.lower():
                return self.objects[obj_id]

        return None

    def get_object_position(self, identifier: str) -> Optional[Tuple[float, float]]:
        """Get object center position."""
        obj = self.resolve_object(identifier)
        return obj.position if obj else None

    def get_object_bbox(self, identifier: str) -> Optional[List[int]]:
        """Get object bounding box [x1, y1, x2, y2]."""
        obj = self.resolve_object(identifier)
        return obj.bbox if obj else None

    def get_all_objects(self) -> List[ObjectInfo]:
        """Get all objects sorted by z-order."""
        return sorted(self.objects.values(), key=lambda o: o.z_order)

    # ========== Spatial Relations ==========

    def is_left_of(self, a: str, b: str) -> bool:
        """Is object A to the left of object B?"""
        obj_a = self.resolve_object(a)
        obj_b = self.resolve_object(b)
        if not obj_a or not obj_b:
            return False
        return obj_a.position[0] < obj_b.position[0] - self.horizontal_threshold

    def is_right_of(self, a: str, b: str) -> bool:
        """Is object A to the right of object B?"""
        obj_a = self.resolve_object(a)
        obj_b = self.resolve_object(b)
        if not obj_a or not obj_b:
            return False
        return obj_a.position[0] > obj_b.position[0] + self.horizontal_threshold

    def is_above(self, a: str, b: str) -> bool:
        """Is object A above object B? (lower y in image coords)"""
        obj_a = self.resolve_object(a)
        obj_b = self.resolve_object(b)
        if not obj_a or not obj_b:
            return False
        return obj_a.position[1] < obj_b.position[1] - self.vertical_threshold

    def is_below(self, a: str, b: str) -> bool:
        """Is object A below object B? (higher y in image coords)"""
        obj_a = self.resolve_object(a)
        obj_b = self.resolve_object(b)
        if not obj_a or not obj_b:
            return False
        return obj_a.position[1] > obj_b.position[1] + self.vertical_threshold

    def is_in_front_of(self, a: str, b: str) -> bool:
        """Is object A in front of object B? (HIGHER z_order = closer to camera in this dataset)"""
        obj_a = self.resolve_object(a)
        obj_b = self.resolve_object(b)
        if not obj_a or not obj_b:
            return False
        # Note: z-order is inverted in this dataset - higher z = visually closer
        return obj_a.z_order > obj_b.z_order

    def is_behind(self, a: str, b: str) -> bool:
        """Is object A behind object B? (LOWER z_order = farther from camera in this dataset)"""
        obj_a = self.resolve_object(a)
        obj_b = self.resolve_object(b)
        if not obj_a or not obj_b:
            return False
        # Note: z-order is inverted in this dataset - lower z = visually farther
        return obj_a.z_order < obj_b.z_order

    def is_near(self, a: str, b: str) -> bool:
        """Are objects A and B near each other?"""
        dist = self.distance_between(a, b)
        return dist is not None and dist < self.proximity_near

    def distance_between(self, a: str, b: str) -> Optional[float]:
        """Euclidean distance between object centers."""
        obj_a = self.resolve_object(a)
        obj_b = self.resolve_object(b)
        if not obj_a or not obj_b:
            return None
        return math.sqrt(
            (obj_a.position[0] - obj_b.position[0])**2 +
            (obj_a.position[1] - obj_b.position[1])**2
        )

    def get_spatial_relation(self, a: str, b: str) -> Dict[str, Any]:
        """Get all spatial relations between A and B."""
        return {
            'left_of': self.is_left_of(a, b),
            'right_of': self.is_right_of(a, b),
            'above': self.is_above(a, b),
            'below': self.is_below(a, b),
            'in_front_of': self.is_in_front_of(a, b),
            'behind': self.is_behind(a, b),
            'near': self.is_near(a, b),
            'distance': self.distance_between(a, b)
        }

    # ========== Spatial Queries ==========

    def get_objects_left_of(self, reference: str) -> List[ObjectInfo]:
        """Get all objects to the left of reference."""
        ref_obj = self.resolve_object(reference)
        if not ref_obj:
            return []
        return [obj for obj in self.objects.values()
                if obj.id != ref_obj.id and self.is_left_of(obj.id, reference)]

    def get_objects_right_of(self, reference: str) -> List[ObjectInfo]:
        """Get all objects to the right of reference."""
        ref_obj = self.resolve_object(reference)
        if not ref_obj:
            return []
        return [obj for obj in self.objects.values()
                if obj.id != ref_obj.id and self.is_right_of(obj.id, reference)]

    def get_objects_behind(self, reference: str) -> List[ObjectInfo]:
        """Get all objects behind reference (farther from camera)."""
        ref_obj = self.resolve_object(reference)
        if not ref_obj:
            return []
        return [obj for obj in self.objects.values()
                if obj.id != ref_obj.id and self.is_behind(obj.id, reference)]

    def get_objects_in_front_of(self, reference: str) -> List[ObjectInfo]:
        """Get all objects in front of reference (closer to camera)."""
        ref_obj = self.resolve_object(reference)
        if not ref_obj:
            return []
        return [obj for obj in self.objects.values()
                if obj.id != ref_obj.id and self.is_in_front_of(obj.id, reference)]

    def get_nearest_object_to(self, reference: str) -> Optional[ObjectInfo]:
        """Get the nearest object to reference."""
        ref_obj = self.resolve_object(reference)
        if not ref_obj:
            return None

        nearest = None
        min_dist = float('inf')

        for obj in self.objects.values():
            if obj.id == ref_obj.id:
                continue
            dist = self.distance_between(obj.id, reference)
            if dist and dist < min_dist:
                min_dist = dist
                nearest = obj

        return nearest

    # ========== Path Validation ==========

    def point_in_bbox(self, point: Tuple[float, float], bbox: List[int]) -> bool:
        """Check if a point is inside a bounding box."""
        x, y = point
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

    def line_intersects_bbox(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        bbox: List[int]
    ) -> bool:
        """Check if a line segment intersects a bounding box."""
        # Simple check: if either endpoint is in bbox, intersects
        if self.point_in_bbox(p1, bbox) or self.point_in_bbox(p2, bbox):
            return True

        # Check if line crosses bbox edges
        x1, y1 = p1
        x2, y2 = p2
        bx1, by1, bx2, by2 = bbox

        # Parametric line intersection with bbox edges
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        def intersect(A, B, C, D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

        # Check against all four edges
        edges = [
            ((bx1, by1), (bx2, by1)),  # top
            ((bx2, by1), (bx2, by2)),  # right
            ((bx2, by2), (bx1, by2)),  # bottom
            ((bx1, by2), (bx1, by1)),  # left
        ]

        for e1, e2 in edges:
            if intersect(p1, p2, e1, e2):
                return True

        return False

    def _shrink_bbox(self, bbox: List[int], shrink_factor: float = 0.3) -> List[int]:
        """
        Shrink a bounding box by a factor (0.3 = shrink by 30% on each side).
        Used for lenient collision detection.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        dx = w * shrink_factor / 2
        dy = h * shrink_factor / 2
        return [
            int(x1 + dx),
            int(y1 + dy),
            int(x2 - dx),
            int(y2 - dy)
        ]

    def is_path_valid(
        self,
        path: List[Tuple[float, float]],
        excluded_objects: List[str] = None,
        lenient_mode: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Check if a path is valid (doesn't pass through obstacles).

        Args:
            path: List of (x, y) waypoints
            excluded_objects: Object IDs to exclude from collision check
                             (e.g., start and end objects)
            lenient_mode: If True, use shrunk bboxes (30% smaller) to allow
                         paths near but not through object centers.
                         This is more realistic for floor-level walking paths.

        Returns:
            Tuple of (is_valid, list of collided object names)
        """
        if not path or len(path) < 2:
            return True, []

        excluded = set(excluded_objects or [])
        collisions = []

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            for obj in self.objects.values():
                if obj.id in excluded or obj.name in excluded:
                    continue

                # Use shrunk bbox in lenient mode
                check_bbox = self._shrink_bbox(obj.bbox) if lenient_mode else obj.bbox

                if self.line_intersects_bbox(p1, p2, check_bbox):
                    if obj.name not in collisions:
                        collisions.append(obj.name)

        return len(collisions) == 0, collisions

    def compute_optimal_path_length(self, start: str, goal: str) -> Optional[float]:
        """
        Compute straight-line distance between two objects.
        (Simple approximation - real optimal path would need pathfinding)
        """
        start_pos = self.get_object_position(start)
        goal_pos = self.get_object_position(goal)
        if not start_pos or not goal_pos:
            return None
        return math.sqrt(
            (goal_pos[0] - start_pos[0])**2 +
            (goal_pos[1] - start_pos[1])**2
        )

    # ========== Perspective Computation ==========

    def get_view_from_position(
        self,
        position: Tuple[float, float],
        facing_angle: float = 0
    ) -> Dict[str, List[ObjectInfo]]:
        """
        Compute what objects are in each direction from a position.

        Args:
            position: (x, y) position
            facing_angle: Direction facing (0=right, 90=down, etc.)

        Returns:
            Dict with keys: 'front', 'behind', 'left', 'right'
        """
        result = {
            'front': [],
            'behind': [],
            'left': [],
            'right': []
        }

        for obj in self.objects.values():
            # Vector from position to object
            dx = obj.position[0] - position[0]
            dy = obj.position[1] - position[1]

            # Angle to object (world frame)
            angle_to_obj = math.degrees(math.atan2(dy, dx))

            # Convert to egocentric (relative to facing)
            relative_angle = (angle_to_obj - facing_angle + 180) % 360 - 180

            # Categorize
            if -45 <= relative_angle < 45:
                result['front'].append(obj)
            elif 45 <= relative_angle < 135:
                result['right'].append(obj)
            elif -135 <= relative_angle < -45:
                result['left'].append(obj)
            else:
                result['behind'].append(obj)

        return result

    # ========== Summary ==========

    def get_scene_summary(self) -> Dict[str, Any]:
        """Get summary of the scene."""
        objects_by_z = {}
        for obj in self.objects.values():
            if obj.z_order not in objects_by_z:
                objects_by_z[obj.z_order] = []
            objects_by_z[obj.z_order].append(obj.name)

        return {
            'image_size': self.image_size,
            'num_objects': len(self.objects),
            'objects': [obj.name for obj in self.get_all_objects()],
            'depth_layers': objects_by_z,
            'z_range': (0, max(objects_by_z.keys()) if objects_by_z else 0)
        }
