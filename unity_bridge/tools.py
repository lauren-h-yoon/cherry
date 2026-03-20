"""
tools.py — LangChain-compatible tool definitions for placing objects in Unity.

These tools wrap the UnityBridge HTTP client so that any VLM with tool-use
support (Claude, GPT-4o, Qwen, …) can manipulate the Unity scene.

Available tools
---------------
  place_object    — Place a labelled sphere at (x, y, z)
  remove_object   — Remove the object nearest to (x, y, z)
  move_object     — Move the object nearest to (x, y, z) to a new position
  clear_scene     — Remove all placed spheres
  get_scene_state — List all placed objects

Coordinate system (Unity, left-handed)
---------------------------------------
  X : Left (−10) … Right (+10)
  Y : Ground (0) … Up (+10)    ← spheres sit on ground at Y = 0.5
  Z : Near  (0)  … Far  (+20)  ← Z=0 at camera, Z=20 is far background

Usage
-----
    from unity_bridge import UnityBridge, create_unity_tools

    bridge = UnityBridge()
    bridge.wait_for_unity()
    tools = create_unity_tools(bridge)

    # Pass `tools` to a LangChain/LangGraph agent or provider.bind_tools()
"""

from typing import List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from .bridge import UnityBridge


# ─── Input schemas ────────────────────────────────────────────────────────────

COORD_NOTE = (
    "Unity scene bounds: X ∈ [−10, 10], Y ∈ [0, 10], Z ∈ [0, 20]. "
    "Ground is Y = 0; a sphere sitting on the ground needs Y = 0.5 (sphere radius). "
    "X negative = left, positive = right. "
    "Z = 0 is at the camera (nearest); Z = 20 is far background. "
    "Y positive = higher up."
)


class PlaceObjectInput(BaseModel):
    """Input schema for place_object."""

    label: str = Field(
        description=(
            "Semantic label for the object being placed (e.g. 'chair', 'table', 'lamp'). "
            "This label appears above the object in the Unity scene."
        )
    )
    x: float = Field(
        description=(
            "Horizontal position along the X axis. "
            "Negative = left of centre, positive = right of centre. "
            f"Range: −10 to +10. {COORD_NOTE}"
        )
    )
    y: float = Field(
        default=0.5,
        description=(
            "Vertical height along the Y axis. "
            "Ground is Y = 0; an object sitting on the ground needs Y = 0.5. "
            "Elevated objects (e.g. a wall painting) use higher values. "
            "Range: 0 to 10."
        )
    )
    z: float = Field(
        description=(
            "Depth position along the Z axis. "
            "Z = 0 is at the camera (nearest); Z = 20 is far background. "
            "Range: 0 to 20."
        )
    )
    scale: float = Field(
        default=1.0,
        description=(
            "Uniform scale of the object. 1.0 is default size. "
            "Use larger values for bigger objects (e.g. 2.0 for a sofa), "
            "smaller for small objects (e.g. 0.5 for a cup)."
        )
    )
    shape: str = Field(
        default="sphere",
        description=(
            "Shape of the object marker. Either 'sphere' or 'cube'. "
            "Use 'cube' for boxy objects (furniture, appliances, buildings), "
            "'sphere' for round or generic objects."
        )
    )


class RemoveObjectInput(BaseModel):
    """Input schema for remove_object."""

    x: float = Field(description=f"X coordinate of the object to remove. {COORD_NOTE}")
    y: float = Field(default=0.5, description="Y coordinate of the object to remove.")
    z: float = Field(description="Z coordinate of the object to remove.")


class MoveObjectInput(BaseModel):
    """Input schema for move_object."""

    x: float = Field(description=f"Current X coordinate of the object to move. {COORD_NOTE}")
    y: float = Field(default=0.5, description="Current Y coordinate of the object to move.")
    z: float = Field(description="Current Z coordinate of the object to move.")
    new_x: float = Field(description="New X coordinate to move the object to.")
    new_y: float = Field(default=0.5, description="New Y coordinate to move the object to.")
    new_z: float = Field(description="New Z coordinate to move the object to.")


class ClearSceneInput(BaseModel):
    """Input schema for clear_scene (no parameters required)."""
    pass


class GetSceneStateInput(BaseModel):
    """Input schema for get_scene_state (no parameters required)."""
    pass


# ─── Tool classes ─────────────────────────────────────────────────────────────

class PlaceObjectTool(BaseTool):
    """
    Place a labelled sphere in the Unity 3D scene to represent an object from
    the image/text description.

    The sphere's position (x, y, z) should encode the spatial relationship of
    the real-world object relative to other objects:
      • Objects to the LEFT  → more negative X
      • Objects to the RIGHT → more positive X
      • Objects in the FOREGROUND (closer) → more negative Z
      • Objects in the BACKGROUND (further away) → more positive Z
      • Objects that are HIGHER UP → more positive Y

    All objects placed as spheres; the label carries semantic meaning.
    """

    name: str = "place_object"
    description: str = (
        "Place a labelled sphere in the Unity 3D scene at position (x, y, z). "
        "Use this to represent each key object from the image in 3D space. "
        "The sphere's position should reflect where the object is spatially "
        "relative to other objects:\n"
        "  • left/right  → X axis (−10 left … +10 right)\n"
        "  • near/far    → Z axis (0 = at camera/nearest … 20 = far background)\n"
        "  • up/down     → Y axis (0 ground … +10 up); use Y=0.5 to sit on the ground\n"
        "Objects are represented as spheres; the label carries the semantic meaning.\n"
        f"{COORD_NOTE}"
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
            resp = self.bridge.place_object(label=label, x=x, y=y, z=z, scale=scale, shape=shape)
            obj_id = resp.get("id", "?")
            return f"Placed '{label}' ({shape}, scale={scale}) at x={x:.2f}, y={y:.2f}, z={z:.2f}."
        except Exception as exc:
            return f"ERROR placing '{label}': {exc}"


class RemoveObjectTool(BaseTool):
    """Remove the object closest to a given (x, y, z) position from the Unity scene."""

    name: str = "remove_object"
    description: str = (
        "Remove the object at the given (x, y, z) position from the Unity scene. "
        "The object closest to the specified coordinates will be removed. "
        f"{COORD_NOTE}"
    )
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
            obj_id = resp.get("id", "?")
            label = resp.get("label", "?")
            return f"Removed '{label}' (id={obj_id}) from x={x:.2f}, y={y:.2f}, z={z:.2f}."
        except Exception as exc:
            return f"ERROR removing object at ({x}, {y}, {z}): {exc}"


class MoveObjectTool(BaseTool):
    """Move the object closest to a given position to a new position in the Unity scene."""

    name: str = "move_object"
    description: str = (
        "Move the object at position (x, y, z) to a new position (new_x, new_y, new_z) "
        "in the Unity scene. The object closest to the source coordinates will be moved. "
        f"{COORD_NOTE}"
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
            obj_id = resp.get("id", "?")
            label = resp.get("label", "?")
            return (
                f"Moved '{label}' (id={obj_id}) from "
                f"({x:.2f}, {y:.2f}, {z:.2f}) to ({new_x:.2f}, {new_y:.2f}, {new_z:.2f})."
            )
        except Exception as exc:
            return f"ERROR moving object at ({x}, {y}, {z}): {exc}"


class ClearSceneTool(BaseTool):
    """Remove all placed spheres from the Unity scene."""

    name: str = "clear_scene"
    description: str = (
        "Remove all placed objects from the Unity scene. "
        "Use this to start over if you want to revise your object placement."
    )
    args_schema: Type[BaseModel] = ClearSceneInput

    bridge: UnityBridge = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, bridge: UnityBridge, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "bridge", bridge)

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            resp = self.bridge.clear_scene()
            removed = resp.get("removed", "?")
            return f"Scene cleared. Removed {removed} object(s)."
        except Exception as exc:
            return f"ERROR clearing scene: {exc}"


class GetSceneStateTool(BaseTool):
    """List all objects currently placed in the Unity scene."""

    name: str = "get_scene_state"
    description: str = (
        "Return a list of all objects currently placed in the Unity scene. "
        "Use this to review and verify your placements before finishing."
    )
    args_schema: Type[BaseModel] = GetSceneStateInput

    bridge: UnityBridge = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, bridge: UnityBridge, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "bridge", bridge)

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            state = self.bridge.get_scene_state()
            return state.summary()
        except Exception as exc:
            return f"ERROR getting scene state: {exc}"


# ─── Embodied perception tools ─────────────────────────────────────────────────

class CaptureViewInput(BaseModel):
    """Input schema for capture_view (no parameters required)."""
    pass


class CaptureViewTool(BaseTool):
    """
    Capture the current Unity camera view as an image.

    Use this to SEE your placed objects and verify the spatial arrangement
    matches your understanding of the original image. This enables:
      - Visual verification of placements
      - Comparison between original image and abstract representation
      - Iterative refinement based on visual feedback
    """

    name: str = "capture_view"
    description: str = (
        "Capture the current Unity camera view as an image. "
        "Use this to visually verify your object placements and ensure "
        "the abstract 3D representation matches the spatial relationships "
        "in the original image. Returns a description of what's visible."
    )
    args_schema: Type[BaseModel] = CaptureViewInput

    bridge: UnityBridge = None
    _last_image: bytes = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, bridge: UnityBridge, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "bridge", bridge)

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            image_data = self.bridge.capture_view()
            object.__setattr__(self, "_last_image", image_data)

            # Get scene state to describe what's visible
            state = self.bridge.get_scene_state()
            if not state.objects:
                return "Captured view: Scene is empty. No objects placed yet."

            return (
                f"Captured view showing {state.count} object(s):\n"
                f"{state.summary()}\n"
                "Use this to verify spatial relationships match the original image."
            )
        except Exception as exc:
            return f"ERROR capturing view: {exc}"


class RotateCameraInput(BaseModel):
    """Input schema for rotate_camera."""

    yaw: float = Field(
        default=0,
        description=(
            "Horizontal rotation in degrees. "
            "Negative = look left, Positive = look right. "
            "Range: -90 to +90 degrees."
        )
    )
    pitch: float = Field(
        default=0,
        description=(
            "Vertical rotation in degrees. "
            "Negative = look down, Positive = look up. "
            "Range: -90 to +90 degrees."
        )
    )


class RotateCameraTool(BaseTool):
    """
    Rotate the Unity camera to view the scene from different angles.

    Use this to verify spatial relationships from multiple viewpoints,
    especially for depth (front/back) relationships that may be ambiguous
    from the default view.
    """

    name: str = "rotate_camera"
    description: str = (
        "Rotate the Unity camera to view the scene from a different angle. "
        "Use this to verify depth relationships (which objects are in front/behind) "
        "by looking at the scene from the side. "
        "yaw: horizontal rotation (-90 left to +90 right), "
        "pitch: vertical rotation (-90 down to +90 up)."
    )
    args_schema: Type[BaseModel] = RotateCameraInput

    bridge: UnityBridge = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, bridge: UnityBridge, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "bridge", bridge)

    def _run(
        self,
        yaw: float = 0,
        pitch: float = 0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            resp = self.bridge.rotate_camera(yaw=yaw, pitch=pitch)
            return (
                f"Camera rotated to yaw={yaw}°, pitch={pitch}°. "
                "Call capture_view to see the scene from this angle."
            )
        except Exception as exc:
            return f"ERROR rotating camera: {exc}"


class ResetCameraInput(BaseModel):
    """Input schema for reset_camera (no parameters required)."""
    pass


class ResetCameraTool(BaseTool):
    """Reset the Unity camera to the default forward-facing orientation."""

    name: str = "reset_camera"
    description: str = (
        "Reset the camera to the default orientation (looking straight ahead). "
        "Use this after rotating to return to the standard viewpoint."
    )
    args_schema: Type[BaseModel] = ResetCameraInput

    bridge: UnityBridge = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, bridge: UnityBridge, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "bridge", bridge)

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            self.bridge.reset_camera()
            return "Camera reset to default orientation (yaw=0, pitch=0)."
        except Exception as exc:
            return f"ERROR resetting camera: {exc}"


# ─── Factories ────────────────────────────────────────────────────────────────

def create_zero_shot_tools(bridge: UnityBridge) -> List[BaseTool]:
    """
    Create the minimal tool set for zero-shot placement: [place_object] only.

    Use this for single-turn evaluations where the model places all objects
    in one response with no follow-up.
    """
    return [PlaceObjectTool(bridge=bridge)]


def create_unity_tools(bridge: UnityBridge, include_camera_tools: bool = False) -> List[BaseTool]:
    """
    Create the full Unity tool set for multi-turn agentic use.

    Returns
    -------
    List[BaseTool]
        Core: [place_object, remove_object, move_object, clear_scene, get_scene_state]
        With camera: + [capture_view, rotate_camera, reset_camera]
    """
    tools = [
        PlaceObjectTool(bridge=bridge),
        RemoveObjectTool(bridge=bridge),
        MoveObjectTool(bridge=bridge),
        ClearSceneTool(bridge=bridge),
        GetSceneStateTool(bridge=bridge),
    ]

    if include_camera_tools:
        tools.extend([
            CaptureViewTool(bridge=bridge),
            RotateCameraTool(bridge=bridge),
            ResetCameraTool(bridge=bridge),
        ])

    return tools
