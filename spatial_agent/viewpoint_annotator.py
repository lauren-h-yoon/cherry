#!/usr/bin/env python3
"""
viewpoint_annotator.py - Annotate images with viewpoint markers for spatial tasks

Provides visual markers for:
- Viewpoint position with facing direction arrow
- Start/end markers for path planning
- Reference object highlighting

Used to create annotated images for clearer task instructions.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, List
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class ViewpointMarker:
    """Represents a viewpoint position and facing direction."""
    position: Tuple[float, float]  # (x, y) in image coordinates
    facing: str  # "camera", "away", "left", "right", or angle in degrees
    label: str = "YOU ARE HERE"

    @property
    def facing_angle(self) -> float:
        """Convert facing direction to angle in degrees (0=right, 90=down)."""
        if self.facing == "camera":
            return 270  # Facing up (toward top of image = toward camera)
        elif self.facing == "away":
            return 90   # Facing down (toward bottom = away from camera)
        elif self.facing == "left":
            return 180  # Facing left
        elif self.facing == "right":
            return 0    # Facing right
        else:
            try:
                return float(self.facing)
            except ValueError:
                return 270  # Default to facing camera


class ViewpointAnnotator:
    """
    Annotates images with viewpoint markers for spatial reasoning tasks.

    Provides methods to add:
    - Viewpoint marker (circle + arrow showing facing direction)
    - Path start/end markers
    - Object highlight boxes
    """

    # Visual settings (base values, will be scaled by image size)
    # These are base values for a 1000px wide image
    BASE_MARKER_RADIUS = 35
    BASE_ARROW_LENGTH = 60
    BASE_ARROW_HEAD_SIZE = 20
    BASE_FONT_SIZE = 28
    BASE_LABEL_PADDING = 8

    # Colors (RGBA)
    COLORS = {
        'viewpoint_fill': (52, 152, 219, 220),      # Blue
        'viewpoint_outline': (41, 128, 185, 255),
        'viewpoint_arrow': (231, 76, 60, 255),       # Red arrow
        'start_fill': (46, 204, 113, 220),           # Green
        'start_outline': (39, 174, 96, 255),
        'end_fill': (241, 196, 15, 220),             # Yellow/Gold
        'end_outline': (243, 156, 18, 255),
        'label_bg': (0, 0, 0, 200),
        'label_text': (255, 255, 255, 255),
        'highlight_box': (155, 89, 182, 180),        # Purple
    }

    def __init__(self, image_path: str):
        """
        Initialize annotator with an image.

        Args:
            image_path: Path to the image to annotate
        """
        self.image_path = image_path
        self.original_image = Image.open(image_path).convert('RGBA')
        self.image_size = self.original_image.size  # (width, height)

        # Scale visual elements based on image size
        # Use the smaller dimension to ensure markers are visible
        scale_factor = min(self.image_size) / 1000.0
        scale_factor = max(scale_factor, 1.0)  # Don't shrink below base size

        self.MARKER_RADIUS = int(self.BASE_MARKER_RADIUS * scale_factor)
        self.ARROW_LENGTH = int(self.BASE_ARROW_LENGTH * scale_factor)
        self.ARROW_HEAD_SIZE = int(self.BASE_ARROW_HEAD_SIZE * scale_factor)
        self.FONT_SIZE = int(self.BASE_FONT_SIZE * scale_factor)
        self.LABEL_PADDING = int(self.BASE_LABEL_PADDING * scale_factor)

        # Create working copy
        self.annotated_image = self.original_image.copy()
        self.overlay = Image.new('RGBA', self.image_size, (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.overlay)

        # Try to load font with scaled size
        try:
            self.font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", self.FONT_SIZE)
            self.font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(self.FONT_SIZE * 0.8))
        except:
            self.font = ImageFont.load_default()
            self.font_small = self.font

    def _draw_arrow(
        self,
        start: Tuple[float, float],
        angle_degrees: float,
        length: float,
        color: Tuple[int, int, int, int],
        head_size: float = None
    ):
        """Draw an arrow from start position in the given direction."""
        head_size = head_size or self.ARROW_HEAD_SIZE

        # Calculate end point
        angle_rad = math.radians(angle_degrees)
        end_x = start[0] + length * math.cos(angle_rad)
        end_y = start[1] + length * math.sin(angle_rad)
        end = (end_x, end_y)

        # Draw arrow shaft
        self.draw.line([start, end], fill=color, width=5)

        # Draw arrow head
        head_angle1 = angle_rad + math.radians(150)
        head_angle2 = angle_rad - math.radians(150)

        head_points = [
            end,
            (end_x + head_size * math.cos(head_angle1),
             end_y + head_size * math.sin(head_angle1)),
            (end_x + head_size * math.cos(head_angle2),
             end_y + head_size * math.sin(head_angle2)),
        ]
        self.draw.polygon(head_points, fill=color)

    def _draw_label(
        self,
        position: Tuple[float, float],
        text: str,
        bg_color: Tuple[int, int, int, int] = None,
        text_color: Tuple[int, int, int, int] = None,
        above: bool = True
    ):
        """Draw a label with background at the given position."""
        bg_color = bg_color or self.COLORS['label_bg']
        text_color = text_color or self.COLORS['label_text']

        # Get text size
        bbox = self.draw.textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position label
        label_x = position[0] - text_width // 2
        if above:
            label_y = position[1] - self.MARKER_RADIUS - text_height - self.LABEL_PADDING * 3
        else:
            label_y = position[1] + self.MARKER_RADIUS + self.LABEL_PADDING

        # Draw background
        self.draw.rectangle(
            [label_x - self.LABEL_PADDING, label_y - self.LABEL_PADDING,
             label_x + text_width + self.LABEL_PADDING, label_y + text_height + self.LABEL_PADDING],
            fill=bg_color
        )

        # Draw text
        self.draw.text((label_x, label_y), text, fill=text_color, font=self.font)

    def add_viewpoint_marker(
        self,
        position: Tuple[float, float],
        facing: str = "camera",
        label: str = "YOU ARE HERE"
    ) -> 'ViewpointAnnotator':
        """
        Add a viewpoint marker showing position and facing direction.

        Args:
            position: (x, y) position in image coordinates
            facing: Direction facing - "camera", "away", "left", "right", or angle
            label: Label text to display

        Returns:
            self for chaining
        """
        marker = ViewpointMarker(position, facing, label)
        x, y = position

        # Draw outer circle (viewpoint position)
        self.draw.ellipse(
            [x - self.MARKER_RADIUS, y - self.MARKER_RADIUS,
             x + self.MARKER_RADIUS, y + self.MARKER_RADIUS],
            fill=self.COLORS['viewpoint_fill'],
            outline=self.COLORS['viewpoint_outline'],
            width=4
        )

        # Draw inner dot
        inner_radius = 8
        self.draw.ellipse(
            [x - inner_radius, y - inner_radius,
             x + inner_radius, y + inner_radius],
            fill=self.COLORS['viewpoint_outline']
        )

        # Draw facing arrow
        self._draw_arrow(
            start=position,
            angle_degrees=marker.facing_angle,
            length=self.ARROW_LENGTH,
            color=self.COLORS['viewpoint_arrow']
        )

        # Draw label
        self._draw_label(position, label, above=True)

        # Draw facing direction text
        facing_text = f"Facing: {facing}"
        bbox = self.draw.textbbox((0, 0), facing_text, font=self.font_small)
        text_width = bbox[2] - bbox[0]
        self.draw.text(
            (x - text_width // 2, y + self.MARKER_RADIUS + self.LABEL_PADDING),
            facing_text,
            fill=self.COLORS['label_text'],
            font=self.font_small
        )

        return self

    def add_start_marker(
        self,
        position: Tuple[float, float],
        label: str = "START"
    ) -> 'ViewpointAnnotator':
        """
        Add a start position marker (for path planning).

        Args:
            position: (x, y) position
            label: Label text

        Returns:
            self for chaining
        """
        x, y = position
        radius = self.MARKER_RADIUS * 0.8

        # Draw circle
        self.draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=self.COLORS['start_fill'],
            outline=self.COLORS['start_outline'],
            width=3
        )

        # Draw "S" in center
        s_bbox = self.draw.textbbox((0, 0), "S", font=self.font)
        s_width = s_bbox[2] - s_bbox[0]
        s_height = s_bbox[3] - s_bbox[1]
        self.draw.text(
            (x - s_width // 2, y - s_height // 2 - 3),
            "S",
            fill=(255, 255, 255, 255),
            font=self.font
        )

        # Draw label
        self._draw_label(
            position, label,
            bg_color=self.COLORS['start_fill'],
            above=True
        )

        return self

    def add_end_marker(
        self,
        position: Tuple[float, float],
        label: str = "GOAL"
    ) -> 'ViewpointAnnotator':
        """
        Add an end/goal position marker (for path planning).

        Args:
            position: (x, y) position
            label: Label text

        Returns:
            self for chaining
        """
        x, y = position
        size = self.MARKER_RADIUS

        # Draw star shape
        points = []
        for i in range(10):
            angle = i * 36 - 90  # Start from top
            r = size if i % 2 == 0 else size * 0.4
            px = x + r * math.cos(math.radians(angle))
            py = y + r * math.sin(math.radians(angle))
            points.append((px, py))

        self.draw.polygon(
            points,
            fill=self.COLORS['end_fill'],
            outline=self.COLORS['end_outline']
        )

        # Draw label
        self._draw_label(
            position, label,
            bg_color=self.COLORS['end_fill'],
            text_color=(0, 0, 0, 255),
            above=True
        )

        return self

    def add_object_highlight(
        self,
        bbox: List[float],
        label: str = None
    ) -> 'ViewpointAnnotator':
        """
        Highlight an object with a bounding box.

        Args:
            bbox: [x1, y1, x2, y2] bounding box
            label: Optional label text

        Returns:
            self for chaining
        """
        x1, y1, x2, y2 = bbox

        # Draw rectangle outline
        self.draw.rectangle(
            [x1, y1, x2, y2],
            outline=self.COLORS['highlight_box'],
            width=4
        )

        # Draw label if provided
        if label:
            center = ((x1 + x2) / 2, y1)
            self._draw_label(
                center, label,
                bg_color=self.COLORS['highlight_box'],
                above=True
            )

        return self

    def get_image(self) -> Image.Image:
        """
        Get the annotated image.

        Returns:
            PIL Image with annotations
        """
        # Composite overlay onto original
        result = Image.alpha_composite(self.original_image, self.overlay)
        return result.convert('RGB')

    def save(self, output_path: str) -> str:
        """
        Save the annotated image.

        Args:
            output_path: Path to save the image

        Returns:
            Path where image was saved
        """
        img = self.get_image()
        img.save(output_path)
        return output_path

    def reset(self) -> 'ViewpointAnnotator':
        """Reset annotations (clear overlay)."""
        self.overlay = Image.new('RGBA', self.image_size, (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.overlay)
        return self


# ============================================================================
# CONVENIENCE FUNCTIONS FOR TASK ANNOTATION
# ============================================================================

def annotate_for_egocentric_task(
    image_path: str,
    viewpoint_position: Tuple[float, float],
    facing: str = "camera",
    output_path: str = None
) -> Image.Image:
    """
    Create annotated image for egocentric tasks.

    Adds viewpoint marker at the specified position.

    Args:
        image_path: Path to original image
        viewpoint_position: Where the viewer is standing
        facing: Direction facing
        output_path: Optional path to save annotated image

    Returns:
        Annotated PIL Image
    """
    annotator = ViewpointAnnotator(image_path)
    annotator.add_viewpoint_marker(viewpoint_position, facing)

    img = annotator.get_image()
    if output_path:
        img.save(output_path)

    return img


def annotate_for_allocentric_path(
    image_path: str,
    start_position: Tuple[float, float],
    end_position: Tuple[float, float],
    start_label: str = "START",
    end_label: str = "GOAL",
    output_path: str = None
) -> Image.Image:
    """
    Create annotated image for allocentric path planning.

    Adds start and end markers.

    Args:
        image_path: Path to original image
        start_position: Start object position
        end_position: Goal object position
        start_label: Label for start
        end_label: Label for goal
        output_path: Optional path to save

    Returns:
        Annotated PIL Image
    """
    annotator = ViewpointAnnotator(image_path)
    annotator.add_start_marker(start_position, start_label)
    annotator.add_end_marker(end_position, end_label)

    img = annotator.get_image()
    if output_path:
        img.save(output_path)

    return img


def annotate_for_allocentric_qa(
    image_path: str,
    viewpoint_position: Tuple[float, float],
    facing: str = "camera",
    reference_bbox: List[float] = None,
    reference_label: str = None,
    target_bbox: List[float] = None,
    target_label: str = None,
    output_path: str = None
) -> Image.Image:
    """
    Create annotated image for allocentric spatial QA.

    Adds viewpoint marker and optionally highlights reference/target objects.

    Args:
        image_path: Path to original image
        viewpoint_position: Viewpoint location
        facing: Direction facing
        reference_bbox: Optional bbox for reference object
        reference_label: Label for reference
        target_bbox: Optional bbox for target object
        target_label: Label for target
        output_path: Optional path to save

    Returns:
        Annotated PIL Image
    """
    annotator = ViewpointAnnotator(image_path)
    annotator.add_viewpoint_marker(viewpoint_position, facing)

    if reference_bbox:
        annotator.add_object_highlight(reference_bbox, reference_label)
    if target_bbox:
        annotator.add_object_highlight(target_bbox, target_label)

    img = annotator.get_image()
    if output_path:
        img.save(output_path)

    return img


# ============================================================================
# VIEWPOINT POSITION HELPERS
# ============================================================================

def get_standard_viewpoints(image_size: Tuple[int, int]) -> dict:
    """
    Get standard viewpoint positions for an image.

    Args:
        image_size: (width, height) of image

    Returns:
        Dict mapping position name to (x, y) coordinates
    """
    w, h = image_size

    return {
        'center': (w / 2, h * 0.5),
        'center_bottom': (w / 2, h * 0.85),  # Bottom-middle (viewer's natural position)
        'bottom_left': (w * 0.25, h * 0.85),
        'bottom_right': (w * 0.75, h * 0.85),
        'center_left': (w * 0.2, h * 0.5),
        'center_right': (w * 0.8, h * 0.5),
    }


def compute_facing_toward_object(
    viewpoint: Tuple[float, float],
    target: Tuple[float, float]
) -> float:
    """
    Compute facing angle to look toward an object.

    Args:
        viewpoint: (x, y) position of viewer
        target: (x, y) position of object to face

    Returns:
        Angle in degrees (0=right, 90=down, 180=left, 270=up)
    """
    dx = target[0] - viewpoint[0]
    dy = target[1] - viewpoint[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle


if __name__ == "__main__":
    # Demo usage
    import argparse

    parser = argparse.ArgumentParser(description="Demo viewpoint annotation")
    parser.add_argument("--image", "-i", required=True, help="Image path")
    parser.add_argument("--output", "-o", default="annotated_viewpoint.png")
    parser.add_argument("--facing", "-f", default="camera",
                       choices=["camera", "away", "left", "right"])

    args = parser.parse_args()

    # Load image to get size
    img = Image.open(args.image)
    w, h = img.size

    # Add viewpoint at center-bottom
    annotator = ViewpointAnnotator(args.image)
    annotator.add_viewpoint_marker(
        position=(w / 2, h * 0.7),
        facing=args.facing,
        label="YOU ARE HERE"
    )

    annotator.save(args.output)
    print(f"Saved annotated image to {args.output}")
