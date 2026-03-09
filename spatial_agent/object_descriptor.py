#!/usr/bin/env python3
"""
object_descriptor.py - Generate human-readable object descriptors

Converts internal object IDs (cabinet_0, stool_1) into natural language
descriptors that can be understood without seeing labels:
- "the leftmost cabinet"
- "the stool in the foreground"
- "the center table"
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .ground_truth import GroundTruth, ObjectInfo


@dataclass
class ObjectDescriptor:
    """Descriptor for a single object."""
    object_id: str
    object_name: str
    category: str
    descriptor: str  # Full descriptor like "the leftmost cabinet"
    short_descriptor: str  # Just the qualifier like "leftmost"
    is_unique: bool  # True if only one of this category in scene

    # Position info for validation
    horizontal_rank: int  # 0 = leftmost
    depth_rank: int  # 0 = frontmost
    total_in_category: int


class ObjectDescriptorGenerator:
    """
    Generates human-readable descriptors for objects in a scene.

    For disambiguation, uses:
    - Horizontal position: leftmost, rightmost, center, second from left, etc.
    - Depth: frontmost, in the background, middle-depth
    - Combined: "the leftmost cabinet in the foreground"
    """

    def __init__(self, ground_truth: GroundTruth):
        self.gt = ground_truth
        self.objects = ground_truth.get_all_objects()

        # Group objects by category
        self.by_category: Dict[str, List[ObjectInfo]] = defaultdict(list)
        for obj in self.objects:
            # Extract base category (cabinet_0 -> cabinet)
            category = self._get_base_category(obj.name)
            self.by_category[category].append(obj)

        # Sort each category by position
        for category in self.by_category:
            self.by_category[category].sort(key=lambda o: o.position[0])

        # Pre-compute descriptors for all objects
        self._descriptors: Dict[str, ObjectDescriptor] = {}
        self._compute_all_descriptors()

    def _get_base_category(self, name: str) -> str:
        """Extract base category from object name (cabinet_0 -> cabinet)."""
        # Handle names like "cabinet_0", "stool_1", "person"
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return name

    def _compute_all_descriptors(self):
        """Compute descriptors for all objects."""
        for category, objects in self.by_category.items():
            n = len(objects)

            # Sort by x-position for horizontal ranking
            by_x = sorted(objects, key=lambda o: o.position[0])
            # Sort by z-order for depth ranking (lower z = closer/front)
            by_z = sorted(objects, key=lambda o: o.z_order)

            for obj in objects:
                h_rank = by_x.index(obj)
                d_rank = by_z.index(obj)

                if n == 1:
                    # Unique object - just use category
                    descriptor = f"the {category}"
                    short_desc = ""
                    is_unique = True
                else:
                    # Need disambiguation
                    short_desc, descriptor = self._get_position_descriptor(
                        obj, category, n, h_rank, d_rank, by_x, by_z
                    )
                    is_unique = False

                # Only add if we have a natural descriptor
                if descriptor is not None:
                    self._descriptors[obj.name] = ObjectDescriptor(
                        object_id=obj.id,
                        object_name=obj.name,
                        category=category,
                        descriptor=descriptor,
                        short_descriptor=short_desc or "",
                        is_unique=is_unique,
                        horizontal_rank=h_rank,
                        depth_rank=d_rank,
                        total_in_category=n
                    )

    def has_natural_descriptor(self, object_name: str) -> bool:
        """Check if an object has a natural (non-ambiguous) descriptor."""
        return object_name in self._descriptors

    def get_describable_objects(self) -> List[str]:
        """Get list of object names that have natural descriptors."""
        return list(self._descriptors.keys())

    def _get_position_descriptor(
        self,
        obj: ObjectInfo,
        category: str,
        total: int,
        h_rank: int,
        d_rank: int,
        by_x: List[ObjectInfo],
        by_z: List[ObjectInfo]
    ) -> Tuple[str, str]:
        """
        Get position-based descriptor for an object.

        Returns:
            Tuple of (short_descriptor, full_descriptor)
            Returns (None, None) if no natural descriptor is possible.
        """
        h_desc = self._horizontal_descriptor(h_rank, total)
        d_desc = self._depth_descriptor(d_rank, total)

        # Only use natural descriptors (leftmost, rightmost, center, foreground, background)
        natural_h_descs = {'leftmost', 'rightmost', 'center'}
        natural_d_descs = {'in the foreground', 'in the background'}

        # Check if horizontal alone is unique and natural
        h_counts = defaultdict(int)
        for i, o in enumerate(by_x):
            h_counts[self._horizontal_descriptor(i, total)] += 1

        if h_desc in natural_h_descs and h_counts[h_desc] == 1:
            # Horizontal descriptor is unique and natural
            return h_desc, f"the {h_desc} {category}"

        # Try combining with depth if both are natural
        if h_desc in natural_h_descs and d_desc in natural_d_descs:
            combined = f"{h_desc} {category} {d_desc}"
            return f"{h_desc}, {d_desc}", f"the {combined}"

        # Try depth alone if natural
        if d_desc in natural_d_descs:
            # Check if depth alone is unique
            d_counts = defaultdict(int)
            for i, o in enumerate(by_z):
                d_counts[self._depth_descriptor(i, total)] += 1
            if d_counts[d_desc] == 1:
                return d_desc, f"the {category} {d_desc}"

        # No natural descriptor possible - mark as ambiguous
        return None, None

    def _horizontal_descriptor(self, rank: int, total: int) -> str:
        """Get horizontal position descriptor."""
        if total <= 1:
            return ""
        if rank == 0:
            return "leftmost"
        if rank == total - 1:
            return "rightmost"
        if total == 3 and rank == 1:
            return "center"
        if total > 3:
            if rank == 1:
                return "second from left"
            if rank == total - 2:
                return "second from right"
            # For middle positions in larger groups
            mid = total // 2
            if rank == mid:
                return "center"
        return f"{self._ordinal(rank + 1)} from left"

    def _depth_descriptor(self, rank: int, total: int) -> str:
        """Get depth position descriptor."""
        if total <= 1:
            return ""
        if rank == 0:
            return "in the foreground"
        if rank == total - 1:
            return "in the background"
        if total >= 3:
            mid = total // 2
            if rank == mid:
                return "at middle depth"
        return ""

    def _ordinal(self, n: int) -> str:
        """Convert number to ordinal (1 -> 1st, 2 -> 2nd, etc.)."""
        if 11 <= n % 100 <= 13:
            suffix = 'th'
        else:
            suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
        return f"{n}{suffix}"

    # ========== Public API ==========

    def get_descriptor(self, object_name: str) -> Optional[ObjectDescriptor]:
        """Get descriptor for an object by name."""
        return self._descriptors.get(object_name)

    def get_descriptor_string(self, object_name: str) -> str:
        """Get the full descriptor string for an object."""
        desc = self._descriptors.get(object_name)
        if desc:
            return desc.descriptor
        return f"the {object_name}"

    def get_all_descriptors(self) -> Dict[str, ObjectDescriptor]:
        """Get all computed descriptors."""
        return self._descriptors.copy()

    def resolve_descriptor(self, descriptor: str) -> Optional[ObjectInfo]:
        """
        Resolve a descriptor back to an object.

        Args:
            descriptor: Natural language like "the leftmost cabinet"

        Returns:
            Matching ObjectInfo or None
        """
        descriptor_lower = descriptor.lower()

        # Check exact matches
        for obj_name, desc in self._descriptors.items():
            if desc.descriptor.lower() == descriptor_lower:
                return self.gt.resolve_object(obj_name)

        # Check partial matches
        for obj_name, desc in self._descriptors.items():
            if desc.descriptor.lower() in descriptor_lower:
                return self.gt.resolve_object(obj_name)
            if descriptor_lower in desc.descriptor.lower():
                return self.gt.resolve_object(obj_name)

        return None

    # ========== Egocentric Descriptors ==========

    def get_egocentric_descriptor(
        self,
        object_name: str,
        viewer_position: Tuple[float, float] = None
    ) -> str:
        """
        Get egocentric descriptor relative to viewer position.

        If viewer_position is None, uses image center as viewer.

        Returns descriptors like:
        - "the cabinet on your left"
        - "the stool in the foreground"
        - "the nearest chair"
        """
        obj = self.gt.resolve_object(object_name)
        if not obj:
            return f"the {object_name}"

        # Default viewer at image center
        if viewer_position is None:
            viewer_position = (
                self.gt.image_width / 2,
                self.gt.image_height / 2
            )

        category = self._get_base_category(obj.name)
        objects_in_cat = self.by_category[category]

        if len(objects_in_cat) == 1:
            return f"the {category}"

        # Compute viewer-relative positions
        def viewer_angle(o):
            dx = o.position[0] - viewer_position[0]
            return dx  # Positive = right, negative = left

        def viewer_distance(o):
            dx = o.position[0] - viewer_position[0]
            dy = o.position[1] - viewer_position[1]
            return (dx**2 + dy**2) ** 0.5

        # Sort by viewer-relative position
        by_angle = sorted(objects_in_cat, key=viewer_angle)
        by_dist = sorted(objects_in_cat, key=viewer_distance)

        angle_rank = by_angle.index(obj)
        dist_rank = by_dist.index(obj)
        n = len(objects_in_cat)

        # Generate egocentric descriptor
        if dist_rank == 0:
            return f"the nearest {category}"
        if dist_rank == n - 1:
            return f"the farthest {category}"
        if angle_rank == 0:
            return f"the {category} on your left"
        if angle_rank == n - 1:
            return f"the {category} on your right"

        # Use depth
        if obj.z_order == min(o.z_order for o in objects_in_cat):
            return f"the {category} in the foreground"
        if obj.z_order == max(o.z_order for o in objects_in_cat):
            return f"the {category} in the background"

        # Fallback to allocentric
        return self.get_descriptor_string(object_name)

    # ========== Directional Queries ==========

    def get_object_in_direction(
        self,
        direction: str,
        category: str = None,
        viewer_position: Tuple[float, float] = None
    ) -> Optional[ObjectInfo]:
        """
        Get object in a given direction from viewer.

        Args:
            direction: "left", "right", "front", "back", "nearest", "farthest"
            category: Optional category filter
            viewer_position: Viewer position (default: image center)

        Returns:
            ObjectInfo of the object in that direction
        """
        if viewer_position is None:
            viewer_position = (
                self.gt.image_width / 2,
                self.gt.image_height / 2
            )

        candidates = self.objects
        if category:
            candidates = self.by_category.get(category, [])

        if not candidates:
            return None

        if direction == "left":
            # Object most to the left of viewer
            left_objects = [o for o in candidates if o.position[0] < viewer_position[0]]
            if left_objects:
                return min(left_objects, key=lambda o: o.position[0])

        elif direction == "right":
            # Object most to the right of viewer
            right_objects = [o for o in candidates if o.position[0] > viewer_position[0]]
            if right_objects:
                return max(right_objects, key=lambda o: o.position[0])

        elif direction == "front" or direction == "foreground":
            # Object closest to camera (lowest z-order)
            return min(candidates, key=lambda o: o.z_order)

        elif direction == "back" or direction == "background":
            # Object farthest from camera (highest z-order)
            return max(candidates, key=lambda o: o.z_order)

        elif direction == "nearest":
            # Object closest to viewer position
            return min(candidates, key=lambda o: (
                (o.position[0] - viewer_position[0])**2 +
                (o.position[1] - viewer_position[1])**2
            ))

        elif direction == "farthest":
            # Object farthest from viewer position
            return max(candidates, key=lambda o: (
                (o.position[0] - viewer_position[0])**2 +
                (o.position[1] - viewer_position[1])**2
            ))

        return None

    def get_objects_in_direction(
        self,
        direction: str,
        viewer_position: Tuple[float, float] = None
    ) -> List[ObjectInfo]:
        """
        Get all objects in a given direction from viewer.

        Args:
            direction: "left", "right", "front", "back"
            viewer_position: Viewer position (default: image center)

        Returns:
            List of objects in that direction
        """
        if viewer_position is None:
            viewer_position = (
                self.gt.image_width / 2,
                self.gt.image_height / 2
            )

        results = []

        for obj in self.objects:
            dx = obj.position[0] - viewer_position[0]

            if direction == "left" and dx < -self.gt.horizontal_threshold:
                results.append(obj)
            elif direction == "right" and dx > self.gt.horizontal_threshold:
                results.append(obj)
            elif direction == "front" or direction == "foreground":
                # Note: z-order appears inverted in this dataset - HIGHER z = closer to camera
                mid_z = sum(o.z_order for o in self.objects) / len(self.objects)
                if obj.z_order > mid_z:
                    results.append(obj)
            elif direction == "back" or direction == "background":
                # LOWER z = farther from camera
                mid_z = sum(o.z_order for o in self.objects) / len(self.objects)
                if obj.z_order < mid_z:
                    results.append(obj)

        return results

    # ========== Summary ==========

    def get_disambiguation_summary(self) -> Dict[str, any]:
        """Get summary of object categories and their descriptors."""
        summary = {}
        for category, objects in self.by_category.items():
            summary[category] = {
                'count': len(objects),
                'is_unique': len(objects) == 1,
                'objects': [
                    {
                        'name': obj.name,
                        'descriptor': self._descriptors[obj.name].descriptor
                    }
                    for obj in objects
                ]
            }
        return summary
