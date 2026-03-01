#!/usr/bin/env python3
"""
state.py - Agent state manager for spatial reasoning

Tracks:
- Current agent position
- Path history
- Visited waypoints
- Waypoint view state (rotation, scale)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import json


@dataclass
class WaypointView:
    """
    Represents the agent's current view of waypoints.

    Can be rotated and scaled to help reason about spatial relationships.
    """
    rotation: float = 0.0  # Degrees, 0 = front view
    scale: float = 1.0     # Zoom factor

    def rotate(self, angle: float) -> 'WaypointView':
        """Rotate the view by given angle (degrees)."""
        new_rotation = (self.rotation + angle) % 360
        return WaypointView(rotation=new_rotation, scale=self.scale)

    def set_scale(self, factor: float) -> 'WaypointView':
        """Set the scale/zoom factor."""
        clamped = max(0.25, min(4.0, factor))  # Clamp between 0.25x and 4x
        return WaypointView(rotation=self.rotation, scale=clamped)

    def get_view_description(self) -> str:
        """Get human-readable description of current view."""
        # Describe rotation
        if self.rotation == 0:
            rot_desc = "front view"
        elif 0 < self.rotation <= 45:
            rot_desc = "slightly rotated right"
        elif 45 < self.rotation <= 90:
            rot_desc = "side view (right)"
        elif 90 < self.rotation <= 135:
            rot_desc = "rotated toward back-right"
        elif 135 < self.rotation <= 180:
            rot_desc = "back view"
        elif 180 < self.rotation <= 225:
            rot_desc = "rotated toward back-left"
        elif 225 < self.rotation <= 270:
            rot_desc = "side view (left)"
        elif 270 < self.rotation <= 315:
            rot_desc = "slightly rotated left"
        else:
            rot_desc = "nearly front view"

        # Describe scale
        if self.scale < 0.5:
            scale_desc = "zoomed out (wide view)"
        elif self.scale < 1.0:
            scale_desc = "slightly zoomed out"
        elif self.scale == 1.0:
            scale_desc = "normal scale"
        elif self.scale <= 2.0:
            scale_desc = "zoomed in"
        else:
            scale_desc = "highly zoomed in (detail view)"

        return f"{rot_desc}, {scale_desc}"

    def transform_position(
        self,
        position: Tuple[float, float],
        depth: float,
        image_center: Tuple[float, float]
    ) -> Tuple[float, float, float]:
        """
        Transform a waypoint position based on current view.

        Returns (x, y, apparent_depth) after rotation and scale.
        """
        x, y = position
        cx, cy = image_center

        # Translate to center
        x_rel = x - cx
        y_rel = y - cy

        # Apply rotation (affects x and depth perception)
        angle_rad = np.radians(self.rotation)

        # Simple 3D-like rotation: x and depth swap as we rotate
        # depth affects how much x shifts
        x_rotated = x_rel * np.cos(angle_rad) - depth * 100 * np.sin(angle_rad)
        apparent_depth = x_rel * np.sin(angle_rad) + depth * np.cos(angle_rad)

        # Apply scale
        x_scaled = x_rotated * self.scale + cx
        y_scaled = (y_rel * self.scale) + cy

        return (x_scaled, y_scaled, apparent_depth)


@dataclass
class AgentState:
    """
    Complete state of the spatial reasoning agent.
    """
    # Current position
    current_waypoint_id: str
    current_z_order: int
    current_position: Tuple[float, float]

    # Target
    target_waypoint_id: str
    target_z_order: int

    # Available waypoints (id -> info)
    waypoints: Dict[str, Dict]

    # Path tracking
    path_history: List[str] = field(default_factory=list)
    move_count: int = 0

    # View state
    view: WaypointView = field(default_factory=WaypointView)

    # Status
    reached_target: bool = False

    def get_current_waypoint_info(self) -> Dict:
        """Get info about current waypoint."""
        return self.waypoints.get(self.current_waypoint_id, {})

    def get_target_waypoint_info(self) -> Dict:
        """Get info about target waypoint."""
        return self.waypoints.get(self.target_waypoint_id, {})

    def get_all_waypoints(self) -> List[Dict]:
        """Get all waypoints sorted by z-order."""
        wps = list(self.waypoints.values())
        return sorted(wps, key=lambda w: w['z_order'])

    def get_waypoints_between(self, z1: int, z2: int) -> List[Dict]:
        """Get waypoints between two z-orders."""
        min_z, max_z = min(z1, z2), max(z1, z2)
        return [
            wp for wp in self.waypoints.values()
            if min_z < wp['z_order'] < max_z
        ]

    def move_to(self, waypoint_id: str) -> Tuple[bool, str]:
        """
        Attempt to move to a waypoint.

        Returns (success, message).
        Note: This doesn't validate occlusion - the agent must reason about that.
        """
        if waypoint_id not in self.waypoints:
            return False, f"Unknown waypoint: {waypoint_id}"

        target_wp = self.waypoints[waypoint_id]

        # Record move
        self.path_history.append(waypoint_id)
        self.move_count += 1

        # Update position
        self.current_waypoint_id = waypoint_id
        self.current_z_order = target_wp['z_order']
        self.current_position = tuple(target_wp['position'])

        # Check if reached target
        if waypoint_id == self.target_waypoint_id:
            self.reached_target = True
            return True, f"Reached target! Moved to {target_wp['name']} (z={target_wp['z_order']})"

        return True, f"Moved to {target_wp['name']} (z={target_wp['z_order']})"

    def rotate_view(self, angle: float) -> str:
        """Rotate the waypoint view."""
        self.view = self.view.rotate(angle)
        return f"Rotated view by {angle}°. Current view: {self.view.get_view_description()}"

    def scale_view(self, factor: float) -> str:
        """Scale/zoom the waypoint view."""
        self.view = self.view.set_scale(factor)
        return f"Set scale to {self.view.scale}x. Current view: {self.view.get_view_description()}"

    def get_status(self) -> str:
        """Get current status summary."""
        current_wp = self.get_current_waypoint_info()
        target_wp = self.get_target_waypoint_info()

        status = f"""
Current Position: {current_wp.get('name', 'unknown')} (z={self.current_z_order})
Target: {target_wp.get('name', 'unknown')} (z={self.target_z_order})
Moves made: {self.move_count}
Path: {' -> '.join(self.path_history) if self.path_history else 'Not started'}
View: {self.view.get_view_description()}
Reached target: {self.reached_target}
"""
        return status.strip()

    def get_path_summary(self) -> Dict:
        """Get summary of the path taken."""
        path_details = []
        for wp_id in self.path_history:
            wp = self.waypoints.get(wp_id, {})
            path_details.append({
                'id': wp_id,
                'name': wp.get('name', 'unknown'),
                'z_order': wp.get('z_order', -1)
            })

        return {
            'path': path_details,
            'total_moves': self.move_count,
            'reached_target': self.reached_target,
            'start_z': self.path_history[0] if self.path_history else None,
            'end_z': self.current_z_order
        }

    def to_dict(self) -> Dict:
        """Serialize state to dictionary."""
        return {
            'current_waypoint_id': self.current_waypoint_id,
            'current_z_order': self.current_z_order,
            'current_position': self.current_position,
            'target_waypoint_id': self.target_waypoint_id,
            'target_z_order': self.target_z_order,
            'waypoints': self.waypoints,
            'path_history': self.path_history,
            'move_count': self.move_count,
            'view_rotation': self.view.rotation,
            'view_scale': self.view.scale,
            'reached_target': self.reached_target
        }

    @classmethod
    def from_scene_config(cls, scene_config) -> 'AgentState':
        """Create agent state from SceneConfig."""
        # Build waypoints dict (include agent and target positions)
        waypoints = {}

        # Add all waypoints
        for wp in scene_config.waypoints:
            waypoints[wp.id] = wp.to_dict()

        # Add agent position
        agent = scene_config.agent_position
        waypoints[agent.id] = agent.to_dict()

        # Add target position
        target = scene_config.target_position
        waypoints[target.id] = target.to_dict()

        return cls(
            current_waypoint_id=agent.id,
            current_z_order=agent.z_order,
            current_position=agent.position,
            target_waypoint_id=target.id,
            target_z_order=target.z_order,
            waypoints=waypoints,
            path_history=[agent.id]  # Start with initial position
        )
