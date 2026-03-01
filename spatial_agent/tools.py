#!/usr/bin/env python3
"""
tools.py - LangChain tools for spatial reasoning agent

Tools:
- get_waypoints: Get all available waypoints with z-order
- move_to: Navigate to a waypoint
- rotate: Rotate waypoint view to reason about depth
- scale: Zoom waypoint view to see depth relationships
"""

from typing import Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from .state import AgentState


class GetWaypointsInput(BaseModel):
    """Input for get_waypoints tool."""
    include_current: bool = Field(
        default=True,
        description="Whether to include current position in the list"
    )


class MoveToInput(BaseModel):
    """Input for move_to tool."""
    waypoint_id: str = Field(
        description="The ID of the waypoint to move to (e.g., 'entity_3')"
    )


class RotateInput(BaseModel):
    """Input for rotate tool."""
    angle: float = Field(
        description="Angle to rotate in degrees. Positive = rotate right, Negative = rotate left"
    )


class ScaleInput(BaseModel):
    """Input for scale tool."""
    factor: float = Field(
        description="Scale factor. >1 zooms in, <1 zooms out. Range: 0.25 to 4.0"
    )


class GetWaypointsTool(BaseTool):
    """Tool to get all available waypoints."""

    name: str = "get_waypoints"
    description: str = """Get all available waypoints in the scene with their z-order (depth).

    Returns a list of waypoints sorted by z-order (z=0 is closest to camera, higher z is farther).
    Each waypoint has:
    - id: unique identifier (use this for move_to)
    - name: descriptive name
    - z_order: depth order (0=closest, higher=farther)
    - relative_depth: normalized depth (0.0=closest, 1.0=farthest)
    - position: (x, y) coordinates in the image

    Use this to understand the spatial layout before planning a path."""

    args_schema: Type[BaseModel] = GetWaypointsInput
    agent_state: AgentState = None

    def __init__(self, agent_state: AgentState, **kwargs):
        super().__init__(**kwargs)
        self.agent_state = agent_state

    def _run(
        self,
        include_current: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Get all waypoints."""
        waypoints = self.agent_state.get_all_waypoints()

        result = "Available waypoints (sorted by z-order, front to back):\n\n"

        for wp in waypoints:
            # Mark current position and target
            markers = []
            if wp['id'] == self.agent_state.current_waypoint_id:
                if not include_current:
                    continue
                markers.append("CURRENT")
            if wp['id'] == self.agent_state.target_waypoint_id:
                markers.append("TARGET")

            marker_str = f" [{', '.join(markers)}]" if markers else ""

            result += f"- {wp['id']}: {wp['name']}{marker_str}\n"
            result += f"    z_order: {wp['z_order']}, depth: {wp['relative_depth']:.3f}\n"
            result += f"    position: ({wp['position'][0]:.0f}, {wp['position'][1]:.0f})\n\n"

        # Add summary
        result += f"\nTotal waypoints: {len(waypoints)}\n"
        result += f"Your position: z={self.agent_state.current_z_order}\n"
        result += f"Target position: z={self.agent_state.target_z_order}\n"

        return result


class MoveToTool(BaseTool):
    """Tool to move to a waypoint."""

    name: str = "move_to"
    description: str = """Move to a specific waypoint.

    IMPORTANT: Before moving, you should reason about whether the path is valid:
    - Consider z-order: moving through objects at intermediate z-orders may be blocked
    - Consider occlusion: objects in front may block access to objects behind
    - Use rotate and scale tools to better understand spatial relationships

    Args:
        waypoint_id: The ID of the waypoint to move to (e.g., 'entity_3')

    Returns success/failure and updates your current position."""

    args_schema: Type[BaseModel] = MoveToInput
    agent_state: AgentState = None

    def __init__(self, agent_state: AgentState, **kwargs):
        super().__init__(**kwargs)
        self.agent_state = agent_state

    def _run(
        self,
        waypoint_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Move to a waypoint."""
        success, message = self.agent_state.move_to(waypoint_id)

        if success:
            result = f"SUCCESS: {message}\n\n"
            result += f"Current status:\n{self.agent_state.get_status()}"
        else:
            result = f"FAILED: {message}"

        return result


class RotateTool(BaseTool):
    """Tool to rotate the waypoint view."""

    name: str = "rotate"
    description: str = """Rotate your view of the waypoints to better understand spatial relationships.

    Rotating helps you:
    - See depth relationships from different angles
    - Understand which objects are behind/in front of others
    - Plan paths that avoid occluded routes

    Args:
        angle: Degrees to rotate. Positive = rotate right, Negative = rotate left.
               Common values: 45, 90, -45, -90, 180

    Returns a description of the new view perspective."""

    args_schema: Type[BaseModel] = RotateInput
    agent_state: AgentState = None

    def __init__(self, agent_state: AgentState, **kwargs):
        super().__init__(**kwargs)
        self.agent_state = agent_state

    def _run(
        self,
        angle: float,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Rotate the view."""
        result = self.agent_state.rotate_view(angle)

        # Provide transformed waypoint positions
        view = self.agent_state.view
        result += "\n\nWaypoints from this perspective:\n"

        # Get image center (approximate from waypoint positions)
        positions = [wp['position'] for wp in self.agent_state.waypoints.values()]
        if positions:
            cx = sum(p[0] for p in positions) / len(positions)
            cy = sum(p[1] for p in positions) / len(positions)
        else:
            cx, cy = 0, 0

        for wp in self.agent_state.get_all_waypoints():
            x, y, apparent_depth = view.transform_position(
                tuple(wp['position']),
                wp['relative_depth'],
                (cx, cy)
            )
            result += f"- {wp['name']} (z={wp['z_order']}): apparent_depth={apparent_depth:.3f}\n"

        return result


class ScaleTool(BaseTool):
    """Tool to scale/zoom the waypoint view."""

    name: str = "scale"
    description: str = """Zoom in or out on the waypoint view to better see depth relationships.

    Scaling helps you:
    - Zoom in to see fine depth differences between nearby waypoints
    - Zoom out to see the overall spatial layout
    - Understand gaps in z-order that might affect path planning

    Args:
        factor: Scale factor.
                >1.0 zooms in (e.g., 2.0 = 2x zoom)
                <1.0 zooms out (e.g., 0.5 = half size)
                Range: 0.25 to 4.0

    Returns a description of the new view scale."""

    args_schema: Type[BaseModel] = ScaleInput
    agent_state: AgentState = None

    def __init__(self, agent_state: AgentState, **kwargs):
        super().__init__(**kwargs)
        self.agent_state = agent_state

    def _run(
        self,
        factor: float,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Scale the view."""
        result = self.agent_state.scale_view(factor)

        # Show depth gaps more clearly at this scale
        result += "\n\nDepth gaps between consecutive waypoints:\n"

        waypoints = self.agent_state.get_all_waypoints()
        for i in range(len(waypoints) - 1):
            wp1, wp2 = waypoints[i], waypoints[i + 1]
            gap = wp2['relative_depth'] - wp1['relative_depth']
            scaled_gap = gap * self.agent_state.view.scale

            # Describe the gap
            if scaled_gap < 0.05:
                gap_desc = "very close"
            elif scaled_gap < 0.15:
                gap_desc = "close"
            elif scaled_gap < 0.3:
                gap_desc = "moderate distance"
            else:
                gap_desc = "far apart"

            result += f"- {wp1['name']} → {wp2['name']}: {gap_desc} (gap={gap:.3f})\n"

        return result


def create_tools(agent_state: AgentState) -> list:
    """Create all tools with the given agent state."""
    return [
        GetWaypointsTool(agent_state=agent_state),
        MoveToTool(agent_state=agent_state),
        RotateTool(agent_state=agent_state),
        ScaleTool(agent_state=agent_state),
    ]
