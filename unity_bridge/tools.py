"""
tools.py — LangChain-compatible tool definitions for placing objects in Unity.

These tools wrap the UnityBridge HTTP client so that any VLM with tool-use
support (Claude, GPT-4o, Qwen, …) can manipulate the Unity scene.

Available tools
---------------
  place_object   — Place a labelled sphere at (x, y, z)
  clear_scene    — Remove all placed spheres
  get_scene_state — List all placed objects

Coordinate system (Unity, left-handed)
---------------------------------------
  X : Left (−10) … Right (+10)
  Y : Ground (0) … Up (+10)      ← spheres sit on ground at Y = 0.5
  Z : Near (−10) … Far  (+10)    ← depth axis away from viewer

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
    "Unity scene bounds: X ∈ [−10, 10], Z ∈ [−10, 10], Y ∈ [0, 10]. "
    "Ground is Y = 0; a sphere sitting on the ground needs Y = 0.5 (sphere radius). "
    "X negative = left, positive = right. "
    "Z negative = near the camera, positive = far from camera (depth). "
    "Y positive = higher up."
)


class PlaceObjectInput(BaseModel):
    """Input schema for place_object."""

    label: str = Field(
        description=(
            "Semantic label for the object being placed (e.g. 'chair', 'table', 'lamp'). "
            "This label appears above the sphere in the Unity scene."
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
            "Set y = 0.5 to place the sphere on the ground (sphere radius = 0.5). "
            "Set y > 0.5 to elevate the object (e.g. y = 2.0 for a hanging lamp). "
            "Range: 0 to 10."
        )
    )
    z: float = Field(
        description=(
            "Depth position along the Z axis. "
            "Negative = closer to the camera (foreground), "
            "positive = further from the camera (background). "
            "Range: −10 to +10."
        )
    )
    color: Optional[str] = Field(
        default=None,
        description=(
            "Optional color for the sphere. "
            "Named colors: 'red', 'blue', 'green', 'yellow', 'purple', 'orange', "
            "'cyan', 'pink', 'white', 'gray', 'brown'. "
            "Also accepts hex strings like '#FF8800'. "
            "If omitted, a distinct color is auto-assigned."
        )
    )
    scale: float = Field(
        default=1.0,
        description=(
            "Size multiplier for the sphere (default 1.0 = diameter 1.0 unit). "
            "Use 1.5–2.0 for large furniture (table, sofa), "
            "0.5–0.8 for small objects (cup, book)."
        )
    )


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
        "  • near/far    → Z axis (−10 near camera … +10 far)\n"
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
        color: Optional[str] = None,
        scale: float = 1.0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            resp = self.bridge.place_object(
                label=label, x=x, y=y, z=z, color=color, scale=scale
            )
            obj_id = resp.get("id", "?")
            color_info = resp.get("color", "auto")
            return (
                f"Placed '{label}' (id={obj_id}) as a sphere at "
                f"x={x:.2f}, y={y:.2f}, z={z:.2f}  |  color={color_info}  |  scale={scale}"
            )
        except Exception as exc:
            return f"ERROR placing '{label}': {exc}"


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


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_unity_tools(bridge: UnityBridge, include_camera_tools: bool = True) -> List[BaseTool]:
    """
    Create all Unity placement tools bound to a given UnityBridge instance.

    Parameters
    ----------
    bridge : UnityBridge
        A connected bridge instance (call bridge.wait_for_unity() first).
    include_camera_tools : bool
        If True, include camera control tools for embodied interaction.

    Returns
    -------
    List[BaseTool]
        Core tools: [place_object, clear_scene, get_scene_state]
        With camera: + [capture_view, rotate_camera, reset_camera]
    """
    tools = [
        PlaceObjectTool(bridge=bridge),
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
