#!/usr/bin/env python3
"""
annotator.py - Generate annotated images with waypoints for spatial reasoning

Overlays:
- Dots (●) for waypoints at entity centers
- Star (★) for target destination
- Triangle (▲) for agent starting position
- Z-order labels for depth understanding
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Waypoint:
    """Represents a waypoint in the scene."""
    id: str
    name: str
    category: str
    position: Tuple[float, float]  # (x, y) center
    z_order: int
    relative_depth: float
    bbox: List[float]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "position": self.position,
            "z_order": self.z_order,
            "relative_depth": self.relative_depth
        }


@dataclass
class SceneConfig:
    """Configuration for a spatial reasoning scenario."""
    image_path: str
    waypoints: List[Waypoint]
    agent_position: Waypoint  # Where agent starts
    target_position: Waypoint  # Where agent needs to go
    image_size: Tuple[int, int]


class SpatialAnnotator:
    """
    Annotates images with waypoints for spatial reasoning tasks.
    """

    # Visual settings
    WAYPOINT_RADIUS = 25
    AGENT_SIZE = 40
    TARGET_SIZE = 45
    FONT_SIZE = 24
    LABEL_OFFSET = 35

    # Colors (RGBA)
    COLORS = {
        'waypoint': (66, 135, 245, 230),      # Blue
        'waypoint_outline': (30, 80, 180, 255),
        'agent': (76, 175, 80, 230),           # Green
        'agent_outline': (40, 120, 40, 255),
        'target': (255, 193, 7, 230),          # Gold/Yellow
        'target_outline': (255, 140, 0, 255),
        'label_bg': (0, 0, 0, 180),
        'label_text': (255, 255, 255, 255),
        'depth_near': (76, 175, 80, 200),      # Green for near
        'depth_far': (244, 67, 54, 200),       # Red for far
    }

    def __init__(self, spatial_graph_path: str, image_path: Optional[str] = None):
        """
        Initialize annotator from spatial graph JSON.

        Args:
            spatial_graph_path: Path to spatial_graph.json
            image_path: Optional override for image path
        """
        with open(spatial_graph_path, 'r') as f:
            self.graph_data = json.load(f)

        self.image_path = image_path or self.graph_data.get('image_path', '')
        self.image_size = tuple(self.graph_data.get('image_size', [0, 0]))

        # Parse waypoints from nodes
        self.waypoints = self._parse_waypoints()

    def _parse_waypoints(self) -> List[Waypoint]:
        """Parse waypoints from spatial graph nodes."""
        waypoints = []
        for node in self.graph_data.get('nodes', []):
            wp = Waypoint(
                id=node['id'],
                name=node['name'],
                category=node['category'],
                position=tuple(node['bbox_center']),
                z_order=node['z_order'],
                relative_depth=node['relative_depth'],
                bbox=node['bbox']
            )
            waypoints.append(wp)
        return waypoints

    def get_waypoint_by_id(self, waypoint_id: str) -> Optional[Waypoint]:
        """Get waypoint by ID."""
        for wp in self.waypoints:
            if wp.id == waypoint_id:
                return wp
        return None

    def get_waypoint_by_z_order(self, z_order: int) -> Optional[Waypoint]:
        """Get waypoint by z-order."""
        for wp in self.waypoints:
            if wp.z_order == z_order:
                return wp
        return None

    def create_scene_config(
        self,
        agent_id: Optional[str] = None,
        agent_z: Optional[int] = None,
        target_id: Optional[str] = None,
        target_z: Optional[int] = None
    ) -> SceneConfig:
        """
        Create a scene configuration with agent and target positions.

        Args:
            agent_id: Entity ID for agent start (or use agent_z)
            agent_z: Z-order for agent start
            target_id: Entity ID for target (or use target_z)
            target_z: Z-order for target
        """
        # Determine agent position
        if agent_id:
            agent_wp = self.get_waypoint_by_id(agent_id)
        elif agent_z is not None:
            agent_wp = self.get_waypoint_by_z_order(agent_z)
        else:
            agent_wp = self.waypoints[0]  # Default to first (closest)

        # Determine target position
        if target_id:
            target_wp = self.get_waypoint_by_id(target_id)
        elif target_z is not None:
            target_wp = self.get_waypoint_by_z_order(target_z)
        else:
            target_wp = self.waypoints[-1]  # Default to last (farthest)

        if not agent_wp or not target_wp:
            raise ValueError("Could not find agent or target waypoint")

        # Other waypoints (excluding agent and target)
        other_waypoints = [
            wp for wp in self.waypoints
            if wp.id != agent_wp.id and wp.id != target_wp.id
        ]

        return SceneConfig(
            image_path=self.image_path,
            waypoints=other_waypoints,
            agent_position=agent_wp,
            target_position=target_wp,
            image_size=self.image_size
        )

    def _get_depth_color(self, relative_depth: float) -> Tuple[int, int, int, int]:
        """Get color based on depth (green=near, red=far)."""
        # Interpolate between green and red
        r = int(76 + (244 - 76) * relative_depth)
        g = int(175 + (67 - 175) * relative_depth)
        b = int(80 + (54 - 80) * relative_depth)
        return (r, g, b, 200)

    def _draw_waypoint(
        self,
        draw: ImageDraw.Draw,
        position: Tuple[float, float],
        z_order: int,
        relative_depth: float,
        radius: int = None
    ):
        """Draw a waypoint dot with z-order label."""
        radius = radius or self.WAYPOINT_RADIUS
        x, y = position

        # Color based on depth
        color = self._get_depth_color(relative_depth)
        outline_color = (max(0, color[0]-40), max(0, color[1]-40), max(0, color[2]-40), 255)

        # Draw filled circle
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=outline_color,
            width=3
        )

        # Draw z-order label
        label = f"z={z_order}"
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", self.FONT_SIZE)
        except:
            font = ImageFont.load_default()

        # Get text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position label above waypoint
        label_x = x - text_width // 2
        label_y = y - radius - self.LABEL_OFFSET

        # Draw label background
        padding = 4
        draw.rectangle(
            [label_x - padding, label_y - padding,
             label_x + text_width + padding, label_y + text_height + padding],
            fill=self.COLORS['label_bg']
        )

        # Draw label text
        draw.text((label_x, label_y), label, fill=self.COLORS['label_text'], font=font)

    def _draw_agent(
        self,
        draw: ImageDraw.Draw,
        position: Tuple[float, float],
        z_order: int
    ):
        """Draw agent marker (triangle pointing up)."""
        x, y = position
        size = self.AGENT_SIZE

        # Triangle points (pointing up)
        points = [
            (x, y - size),           # Top
            (x - size * 0.8, y + size * 0.6),  # Bottom left
            (x + size * 0.8, y + size * 0.6),  # Bottom right
        ]

        draw.polygon(points, fill=self.COLORS['agent'], outline=self.COLORS['agent_outline'])

        # Draw "AGENT" label
        label = f"AGENT (z={z_order})"
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", self.FONT_SIZE)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        label_x = x - text_width // 2
        label_y = y + size + 10

        padding = 4
        draw.rectangle(
            [label_x - padding, label_y - padding,
             label_x + text_width + padding, label_y + text_height + padding],
            fill=(76, 175, 80, 220)
        )
        draw.text((label_x, label_y), label, fill=(255, 255, 255, 255), font=font)

    def _draw_target(
        self,
        draw: ImageDraw.Draw,
        position: Tuple[float, float],
        z_order: int
    ):
        """Draw target marker (star)."""
        x, y = position
        size = self.TARGET_SIZE

        # 5-pointed star
        points = []
        for i in range(10):
            angle = i * 36 - 90  # Start from top
            r = size if i % 2 == 0 else size * 0.4
            px = x + r * np.cos(np.radians(angle))
            py = y + r * np.sin(np.radians(angle))
            points.append((px, py))

        draw.polygon(points, fill=self.COLORS['target'], outline=self.COLORS['target_outline'])

        # Draw "TARGET" label
        label = f"TARGET (z={z_order})"
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", self.FONT_SIZE)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        label_x = x - text_width // 2
        label_y = y - size - self.LABEL_OFFSET - 10

        padding = 4
        draw.rectangle(
            [label_x - padding, label_y - padding,
             label_x + text_width + padding, label_y + text_height + padding],
            fill=(255, 193, 7, 220)
        )
        draw.text((label_x, label_y), label, fill=(0, 0, 0, 255), font=font)

    def annotate(
        self,
        scene_config: SceneConfig,
        output_path: Optional[str] = None,
        show_depth_colors: bool = True
    ) -> Image.Image:
        """
        Generate annotated image with waypoints, agent, and target.

        Args:
            scene_config: Scene configuration
            output_path: Optional path to save annotated image
            show_depth_colors: Color waypoints by depth

        Returns:
            Annotated PIL Image
        """
        # Load original image
        img = Image.open(scene_config.image_path).convert('RGBA')

        # Create overlay for annotations
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw waypoints (sorted by z-order, back to front for proper layering)
        sorted_waypoints = sorted(scene_config.waypoints, key=lambda w: -w.z_order)
        for wp in sorted_waypoints:
            self._draw_waypoint(draw, wp.position, wp.z_order, wp.relative_depth)

        # Draw target (star)
        self._draw_target(
            draw,
            scene_config.target_position.position,
            scene_config.target_position.z_order
        )

        # Draw agent (triangle) - draw last so it's on top
        self._draw_agent(
            draw,
            scene_config.agent_position.position,
            scene_config.agent_position.z_order
        )

        # Composite overlay onto image
        annotated = Image.alpha_composite(img, overlay)

        # Convert to RGB for saving as JPEG/PNG
        annotated_rgb = annotated.convert('RGB')

        if output_path:
            annotated_rgb.save(output_path)
            print(f"Saved annotated image to {output_path}")

        return annotated_rgb

    def generate_scenario(
        self,
        agent_z: int,
        target_z: int,
        output_path: Optional[str] = None
    ) -> Tuple[Image.Image, SceneConfig]:
        """
        Generate a complete scenario with annotated image.

        Args:
            agent_z: Z-order for agent starting position
            target_z: Z-order for target
            output_path: Optional path to save image

        Returns:
            Tuple of (annotated_image, scene_config)
        """
        config = self.create_scene_config(agent_z=agent_z, target_z=target_z)
        img = self.annotate(config, output_path)
        return img, config


def main():
    """Demo: Generate annotated image from spatial graph."""
    import argparse

    parser = argparse.ArgumentParser(description="Annotate image with waypoints")
    parser.add_argument("--graph", "-g", required=True, help="Path to spatial_graph.json")
    parser.add_argument("--image", "-i", help="Override image path")
    parser.add_argument("--output", "-o", default="annotated_scene.png", help="Output path")
    parser.add_argument("--agent-z", type=int, default=0, help="Agent starting z-order")
    parser.add_argument("--target-z", type=int, default=None, help="Target z-order (default: max)")

    args = parser.parse_args()

    # Create annotator
    annotator = SpatialAnnotator(args.graph, args.image)

    # Determine target z (default to farthest)
    if args.target_z is None:
        args.target_z = max(wp.z_order for wp in annotator.waypoints)

    # Generate scenario
    img, config = annotator.generate_scenario(
        agent_z=args.agent_z,
        target_z=args.target_z,
        output_path=args.output
    )

    print(f"\nScenario Generated:")
    print(f"  Agent: {config.agent_position.name} (z={config.agent_position.z_order})")
    print(f"  Target: {config.target_position.name} (z={config.target_position.z_order})")
    print(f"  Waypoints: {len(config.waypoints)}")
    for wp in sorted(config.waypoints, key=lambda w: w.z_order):
        print(f"    z={wp.z_order}: {wp.name}")


if __name__ == "__main__":
    main()
