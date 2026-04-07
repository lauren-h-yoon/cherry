"""
tools.py — LangChain-compatible tool definitions for the Unity bridge pipeline.

Available tools
---------------
  place_object  — Place a labelled object at (x, y, z)
  remove_object — Remove the object nearest to (x, y, z)
  move_object   — Move the object nearest to (x, y, z) to a new position

Coordinate system (Unity, left-handed)
---------------------------------------
  X : Left (−10) … Right (+10)
  Y : Ground (0) … Up (+10)   — objects on ground use Y = 0.5
  Z : Near (0)   … Far (+20)  — Z=0 at camera, Z=20 is far background

Usage
-----
    from unity_bridge import UnityBridge, create_unity_tools

    bridge = UnityBridge()
    bridge.wait_for_unity()
    tools = create_unity_tools(bridge)
"""

from typing import List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from .bridge import UnityBridge


# ─── Input schemas ────────────────────────────────────────────────────────────

class PlaceObjectInput(BaseModel):
    label: str = Field(description="Semantic label for the object (e.g. 'chair', 'table').")
    x: float = Field(description="Horizontal position. Range: -10 (left) to +10 (right).")
    y: float = Field(default=0.5, description="Vertical height. 0=floor, 0.5=ground level. Range: 0-10.")
    z: float = Field(description="Depth position. 0=near camera, 20=far background. Range: 0-20.")
    scale: float = Field(default=1.0, description="Uniform scale multiplier. Default 1.0.")
    shape: str = Field(default="sphere", description="Shape marker: 'sphere' or 'cube'.")


class RemoveObjectInput(BaseModel):
    x: float = Field(description="X coordinate of the object to remove.")
    y: float = Field(default=0.5, description="Y coordinate of the object to remove.")
    z: float = Field(description="Z coordinate of the object to remove.")


class MoveObjectInput(BaseModel):
    x: float = Field(description="Current X coordinate of the object to move.")
    y: float = Field(default=0.5, description="Current Y coordinate of the object to move.")
    z: float = Field(description="Current Z coordinate of the object to move.")
    new_x: float = Field(description="New X coordinate.")
    new_y: float = Field(default=0.5, description="New Y coordinate.")
    new_z: float = Field(description="New Z coordinate.")


# ─── Tool classes ─────────────────────────────────────────────────────────────

class PlaceObjectTool(BaseTool):
    name: str = "place_object"
    description: str = (
        "Place a labelled object in the Unity 3D scene at (x, y, z). "
        "X: -10 (left) to +10 (right). Z: 0 (near) to 20 (far). "
        "Y: 0=floor, 0.5=ground level."
    )
    args_schema: Type[BaseModel] = PlaceObjectInput

    bridge: UnityBridge = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, bridge: UnityBridge, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "bridge", bridge)

    def _run(
        self,
        label: str,
        x: float,
        z: float,
        y: float = 0.5,
        scale: float = 1.0,
        shape: str = "sphere",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            unity_shape = "sphere" if shape == "ball" else shape
            self.bridge.place_object(label=label, x=x, y=y, z=z, scale=scale, shape=unity_shape)
            return f"Placed '{label}' ({shape}, scale={scale}) at ({x:.2f}, {y:.2f}, {z:.2f})."
        except Exception as exc:
            return f"ERROR placing '{label}': {exc}"


class RemoveObjectTool(BaseTool):
    name: str = "remove_object"
    description: str = "Remove the object closest to (x, y, z) from the Unity scene."
    args_schema: Type[BaseModel] = RemoveObjectInput

    bridge: UnityBridge = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, bridge: UnityBridge, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "bridge", bridge)

    def _run(
        self,
        x: float,
        z: float,
        y: float = 0.5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            resp = self.bridge.remove_object(x=x, y=y, z=z)
            return f"Removed '{resp.get('label', '?')}' from ({x:.2f}, {y:.2f}, {z:.2f})."
        except Exception as exc:
            return f"ERROR removing object at ({x}, {y}, {z}): {exc}"


class MoveObjectTool(BaseTool):
    name: str = "move_object"
    description: str = (
        "Move the object closest to (x, y, z) to (new_x, new_y, new_z) in the Unity scene."
    )
    args_schema: Type[BaseModel] = MoveObjectInput

    bridge: UnityBridge = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, bridge: UnityBridge, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "bridge", bridge)

    def _run(
        self,
        x: float,
        z: float,
        new_x: float,
        new_z: float,
        y: float = 0.5,
        new_y: float = 0.5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            resp = self.bridge.move_object(x=x, y=y, z=z, new_x=new_x, new_y=new_y, new_z=new_z)
            return (
                f"Moved '{resp.get('label', '?')}' from "
                f"({x:.2f}, {y:.2f}, {z:.2f}) to ({new_x:.2f}, {new_y:.2f}, {new_z:.2f})."
            )
        except Exception as exc:
            return f"ERROR moving object at ({x}, {y}, {z}): {exc}"


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_unity_tools(bridge: UnityBridge) -> List[BaseTool]:
    """Return the three Unity tools: place_object, remove_object, move_object."""
    return [
        PlaceObjectTool(bridge=bridge),
        RemoveObjectTool(bridge=bridge),
        MoveObjectTool(bridge=bridge),
    ]
