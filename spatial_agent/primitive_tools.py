#!/usr/bin/env python3
"""
primitive_tools.py - Output mechanism tools for embodied spatial intelligence

These tools allow the VLM to EXPRESS its spatial understanding through actions:
- draw_path: Express planned routes
- point_to: Indicate locations
- mark_region: Identify objects/areas
- move_to: Execute navigation
- rotate: Change orientation
- look_at: Direct attention

Tools record VLM outputs for evaluation against ground truth.
"""

import math
from typing import List, Tuple, Optional, Type, Dict, Any
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


# ========== Output Recording ==========

@dataclass
class RecordedOutputs:
    """Stores all VLM outputs for evaluation."""
    paths: List[Dict[str, Any]] = field(default_factory=list)
    points: List[Dict[str, Any]] = field(default_factory=list)
    regions: List[Dict[str, Any]] = field(default_factory=list)
    movements: List[Dict[str, Any]] = field(default_factory=list)
    rotations: List[float] = field(default_factory=list)
    gaze: List[Dict[str, Any]] = field(default_factory=list)

    def clear(self):
        """Clear all recorded outputs."""
        self.paths.clear()
        self.points.clear()
        self.regions.clear()
        self.movements.clear()
        self.rotations.clear()
        self.gaze.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'paths': self.paths,
            'points': self.points,
            'regions': self.regions,
            'movements': self.movements,
            'rotations': self.rotations,
            'gaze': self.gaze
        }


@dataclass
class EmbodiedState:
    """Tracks the agent's embodied state (position, orientation)."""
    position: Tuple[float, float] = (0, 0)
    facing_angle: float = 0.0  # degrees, 0=right, 90=down, 180=left, 270=up

    def move_to(self, x: float, y: float):
        """Update position."""
        self.position = (x, y)

    def rotate(self, angle: float):
        """Rotate by angle degrees (positive = clockwise)."""
        self.facing_angle = (self.facing_angle + angle) % 360

    def face_toward(self, target: Tuple[float, float]):
        """Face toward a target position."""
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        self.facing_angle = math.degrees(math.atan2(dy, dx)) % 360


# ========== Input Schemas ==========

class DrawPathInput(BaseModel):
    """Input for draw_path tool."""
    points: List[List[int]] = Field(
        description="List of [x, y] coordinate pairs as JSON arrays (NOT tuples). Example: [[100, 200], [150, 250], [200, 300]]. First point is start, last is destination."
    )
    label: Optional[str] = Field(
        default=None,
        description="Optional label describing this path (e.g., 'path to table')"
    )


class PointToInput(BaseModel):
    """Input for point_to tool."""
    x: int = Field(description="X coordinate (horizontal position, 0=left edge)")
    y: int = Field(description="Y coordinate (vertical position, 0=top edge)")
    label: Optional[str] = Field(
        default=None,
        description="Optional label for what you're pointing at"
    )


class MarkRegionInput(BaseModel):
    """Input for mark_region tool."""
    x1: int = Field(description="Left edge x coordinate")
    y1: int = Field(description="Top edge y coordinate")
    x2: int = Field(description="Right edge x coordinate")
    y2: int = Field(description="Bottom edge y coordinate")
    label: str = Field(description="Label for what this region contains")


class MoveToInput(BaseModel):
    """Input for move_to tool."""
    x: int = Field(description="Target x coordinate")
    y: int = Field(description="Target y coordinate")


class RotateInput(BaseModel):
    """Input for rotate tool."""
    angle: float = Field(
        description="Degrees to rotate. Positive=clockwise/right, Negative=counter-clockwise/left"
    )


class LookAtInput(BaseModel):
    """Input for look_at tool."""
    x: int = Field(description="X coordinate to look at")
    y: int = Field(description="Y coordinate to look at")


# ========== Primitive Tools ==========

class DrawPathTool(BaseTool):
    """Tool for VLM to draw/express a planned path."""

    name: str = "draw_path"
    description: str = """Draw a path through the scene to express your planned route.

Use this when asked to plan a path, navigate, or show how you would move through the scene.

Args:
    points: List of [x, y] coordinates as JSON arrays (use square brackets, NOT parentheses).
            First point = where you start
            Last point = where you want to reach
            Intermediate points = waypoints along the route
    label: Optional description of this path

IMPORTANT: Use JSON array format with square brackets:
    CORRECT: "points": [[100, 500], [200, 400], [300, 300], [400, 200]]
    WRONG: "points": [(100, 500), (200, 400)]  <-- NO parentheses!

The path will be recorded and evaluated against the actual scene layout.
"""

    args_schema: Type[BaseModel] = DrawPathInput
    recorded_outputs: RecordedOutputs = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, recorded_outputs: RecordedOutputs, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'recorded_outputs', recorded_outputs)

    def _run(
        self,
        points: List[List[int]],
        label: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        if not points or len(points) < 2:
            return "Error: Path must have at least 2 points (start and end)"

        # Convert to tuples for internal processing
        points = [tuple(p) if isinstance(p, list) else p for p in points]

        path_record = {
            'points': points,
            'label': label,
            'num_waypoints': len(points),
            'start': points[0],
            'end': points[-1]
        }
        self.recorded_outputs.paths.append(path_record)

        # Compute path length for feedback
        total_length = sum(
            math.sqrt((points[i+1][0] - points[i][0])**2 +
                     (points[i+1][1] - points[i][1])**2)
            for i in range(len(points) - 1)
        )

        response = f"Path drawn with {len(points)} waypoints.\n"
        response += f"  Start: ({points[0][0]}, {points[0][1]})\n"
        response += f"  End: ({points[-1][0]}, {points[-1][1]})\n"
        response += f"  Total length: {total_length:.0f} units"
        if label:
            response += f"\n  Label: {label}"

        return response


class PointToTool(BaseTool):
    """Tool for VLM to point to/indicate a location."""

    name: str = "point_to"
    description: str = """Point to a specific location in the image.

Use this when asked:
- "Where is the table?" - point to where you see the table
- "Point to what's behind X" - point to that location
- "Indicate the object on the left" - point to it

Args:
    x: Horizontal position (0 = left edge of image)
    y: Vertical position (0 = top edge of image)
    label: What you're pointing at (optional but recommended)

Example:
    point_to(x=300, y=400, label="table")

Your pointing will be evaluated against the actual object positions.
"""

    args_schema: Type[BaseModel] = PointToInput
    recorded_outputs: RecordedOutputs = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, recorded_outputs: RecordedOutputs, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'recorded_outputs', recorded_outputs)

    def _run(
        self,
        x: int,
        y: int,
        label: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        point_record = {
            'x': x,
            'y': y,
            'label': label
        }
        self.recorded_outputs.points.append(point_record)

        response = f"Pointed to ({x}, {y})"
        if label:
            response += f" - {label}"

        return response


class MarkRegionTool(BaseTool):
    """Tool for VLM to mark/highlight a region."""

    name: str = "mark_region"
    description: str = """Mark a rectangular region in the image.

Use this when asked to:
- Identify where an object is located
- Highlight an area of interest
- Show the bounding box of something

Args:
    x1, y1: Top-left corner coordinates
    x2, y2: Bottom-right corner coordinates
    label: What this region contains

Example:
    mark_region(x1=200, y1=150, x2=400, y2=350, label="table")

Your marked region will be compared against actual object bounding boxes.
"""

    args_schema: Type[BaseModel] = MarkRegionInput
    recorded_outputs: RecordedOutputs = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, recorded_outputs: RecordedOutputs, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'recorded_outputs', recorded_outputs)

    def _run(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        label: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        region_record = {
            'bbox': [x1, y1, x2, y2],
            'label': label,
            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
            'area': abs(x2 - x1) * abs(y2 - y1)
        }
        self.recorded_outputs.regions.append(region_record)

        return f"Marked region [{x1}, {y1}, {x2}, {y2}] as '{label}'"


class MoveToTool(BaseTool):
    """Tool for VLM to move to a position (embodied action)."""

    name: str = "move_to"
    description: str = """Move to a specific position in the scene.

Use this when navigating step by step through the scene.
Each move will update your position and be recorded.

Args:
    x: Target x coordinate
    y: Target y coordinate

Your movements will be evaluated for validity (avoiding obstacles, etc.)
"""

    args_schema: Type[BaseModel] = MoveToInput
    recorded_outputs: RecordedOutputs = None
    embodied_state: EmbodiedState = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        recorded_outputs: RecordedOutputs,
        embodied_state: EmbodiedState,
        **kwargs
    ):
        super().__init__(**kwargs)
        object.__setattr__(self, 'recorded_outputs', recorded_outputs)
        object.__setattr__(self, 'embodied_state', embodied_state)

    def _run(
        self,
        x: int,
        y: int,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        old_pos = self.embodied_state.position
        self.embodied_state.move_to(x, y)

        distance = math.sqrt((x - old_pos[0])**2 + (y - old_pos[1])**2)

        movement_record = {
            'from': old_pos,
            'to': (x, y),
            'distance': distance
        }
        self.recorded_outputs.movements.append(movement_record)

        return f"Moved from ({old_pos[0]:.0f}, {old_pos[1]:.0f}) to ({x}, {y}). Distance: {distance:.0f} units."


class RotateTool(BaseTool):
    """Tool for VLM to rotate/change facing direction."""

    name: str = "rotate"
    description: str = """Rotate to change your facing direction.

Use this when you need to:
- Turn to face a different direction
- Reorient yourself in the scene
- Change your perspective

Args:
    angle: Degrees to rotate
           Positive = clockwise (turn right)
           Negative = counter-clockwise (turn left)

Example:
    rotate(angle=90)  # Turn right 90 degrees
    rotate(angle=-45) # Turn left 45 degrees
"""

    args_schema: Type[BaseModel] = RotateInput
    recorded_outputs: RecordedOutputs = None
    embodied_state: EmbodiedState = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        recorded_outputs: RecordedOutputs,
        embodied_state: EmbodiedState,
        **kwargs
    ):
        super().__init__(**kwargs)
        object.__setattr__(self, 'recorded_outputs', recorded_outputs)
        object.__setattr__(self, 'embodied_state', embodied_state)

    def _run(
        self,
        angle: float,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        old_facing = self.embodied_state.facing_angle
        self.embodied_state.rotate(angle)
        new_facing = self.embodied_state.facing_angle

        self.recorded_outputs.rotations.append(angle)

        return f"Rotated {angle:+.0f}°. Was facing {old_facing:.0f}°, now facing {new_facing:.0f}°."


class LookAtTool(BaseTool):
    """Tool for VLM to direct attention/gaze."""

    name: str = "look_at"
    description: str = """Turn to look at a specific location.

Use this to indicate where your attention/gaze is directed,
or to face toward a specific point in the scene.

Args:
    x: X coordinate to look at
    y: Y coordinate to look at
"""

    args_schema: Type[BaseModel] = LookAtInput
    recorded_outputs: RecordedOutputs = None
    embodied_state: EmbodiedState = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        recorded_outputs: RecordedOutputs,
        embodied_state: EmbodiedState,
        **kwargs
    ):
        super().__init__(**kwargs)
        object.__setattr__(self, 'recorded_outputs', recorded_outputs)
        object.__setattr__(self, 'embodied_state', embodied_state)

    def _run(
        self,
        x: int,
        y: int,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        old_facing = self.embodied_state.facing_angle
        self.embodied_state.face_toward((x, y))
        new_facing = self.embodied_state.facing_angle

        gaze_record = {
            'target': (x, y),
            'from_position': self.embodied_state.position,
            'facing_angle': new_facing
        }
        self.recorded_outputs.gaze.append(gaze_record)

        return f"Now looking at ({x}, {y}). Facing angle: {new_facing:.0f}° (was {old_facing:.0f}°)."


# ========== Tool Factory ==========

def create_primitive_tools(
    recorded_outputs: RecordedOutputs = None,
    embodied_state: EmbodiedState = None
) -> Tuple[List[BaseTool], RecordedOutputs, EmbodiedState]:
    """
    Create all primitive tools for embodied spatial intelligence.

    Args:
        recorded_outputs: Optional existing RecordedOutputs (creates new if None)
        embodied_state: Optional existing EmbodiedState (creates new if None)

    Returns:
        Tuple of (tools list, recorded_outputs, embodied_state)
    """
    if recorded_outputs is None:
        recorded_outputs = RecordedOutputs()
    if embodied_state is None:
        embodied_state = EmbodiedState()

    tools = [
        DrawPathTool(recorded_outputs=recorded_outputs),
        PointToTool(recorded_outputs=recorded_outputs),
        MarkRegionTool(recorded_outputs=recorded_outputs),
        MoveToTool(recorded_outputs=recorded_outputs, embodied_state=embodied_state),
        RotateTool(recorded_outputs=recorded_outputs, embodied_state=embodied_state),
        LookAtTool(recorded_outputs=recorded_outputs, embodied_state=embodied_state),
    ]

    return tools, recorded_outputs, embodied_state
