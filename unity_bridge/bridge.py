"""
bridge.py — Python HTTP client for the CherryUnityBridge C# server.

The C# script (CherryUnityBridge.cs) must be attached to a GameObject in your
Unity scene and the scene must be running.  By default it listens on
http://localhost:5555/.

Unity Coordinate System
-----------------------
  X  :  Left (−) / Right (+)
  Y  :  Down (−) / Up (+)   — ground plane at Y = 0
  Z  :  Near (−) / Far  (+) — depth axis

Scene extents:  X ∈ [−10, 10],  Z ∈ [−10, 10],  Y ∈ [0, 10]
Sphere radius:  0.5 Unity units  → sit on ground → set Y = 0.5

Objects in the real scene (chairs, tables, …) are represented as labelled
spheres; the *label* carries the semantic meaning.
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

DEFAULT_UNITY_URL = "http://localhost:5555"


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class PlacedObject:
    """A sphere that has been placed in the Unity scene."""
    id: int
    label: str
    x: float
    y: float
    z: float
    color: str
    scale: float

    @property
    def position(self) -> tuple:
        return (self.x, self.y, self.z)

    def __repr__(self) -> str:
        return f"PlacedObject(id={self.id}, label='{self.label}', pos=({self.x},{self.y},{self.z}))"


@dataclass
class SceneState:
    """Current state of all placed objects in Unity."""
    objects: List[PlacedObject] = field(default_factory=list)
    count: int = 0

    def get_by_label(self, label: str) -> List[PlacedObject]:
        label_lower = label.lower()
        return [o for o in self.objects if label_lower in o.label.lower()]

    def summary(self) -> str:
        if not self.objects:
            return "Scene is empty."
        lines = [f"  [{o.id}] '{o.label}' at ({o.x:.2f}, {o.y:.2f}, {o.z:.2f})"
                 for o in self.objects]
        return f"{self.count} object(s):\n" + "\n".join(lines)


# ─── Bridge client ────────────────────────────────────────────────────────────

class UnityBridge:
    """
    HTTP client that talks to the CherryUnityBridge running inside Unity.

    All methods block until Unity confirms the action.

    Quick-start
    -----------
    bridge = UnityBridge()
    bridge.wait_for_unity()          # blocks until Unity is ready
    bridge.initialize_scene()        # optional re-init
    bridge.place_object("chair", x=1, y=0.5, z=2)
    bridge.place_object("table", x=0, y=0.5, z=0, color="blue")
    state = bridge.get_scene_state()
    print(state.summary())
    bridge.clear_scene()
    """

    def __init__(self, base_url: str = DEFAULT_UNITY_URL):
        self.base_url = base_url.rstrip("/")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _request(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        method: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{path}"
        body = json.dumps(data).encode("utf-8") if data is not None else b"{}"
        method = method or ("POST" if data is not None else "GET")

        req = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Unity at {self.base_url}. "
                "Make sure Unity is running and CherryUnityBridge.cs is attached to "
                f"a scene GameObject.  Original error: {exc}"
            ) from exc

    def _check_error(self, resp: Dict) -> Dict:
        if "error" in resp:
            raise RuntimeError(f"Unity error: {resp['error']}")
        return resp

    # ── Public API ───────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if Unity bridge is reachable and scene is ready."""
        try:
            resp = self._request("health", method="GET")
            return resp.get("status") == "ok"
        except ConnectionError:
            return False

    def wait_for_unity(self, timeout_s: float = 30.0, poll_interval: float = 1.0) -> None:
        """Block until Unity is reachable, polling every poll_interval seconds."""
        import time
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.health_check():
                print("[UnityBridge] Connected to Unity.")
                return
            print(f"[UnityBridge] Waiting for Unity at {self.base_url}…")
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Unity bridge not reachable after {timeout_s}s. "
            "Is the scene running with CherryUnityBridge attached?"
        )

    def initialize_scene(self) -> Dict:
        """Re-initialize the Unity scene (clears all placed objects)."""
        resp = self._check_error(self._request("initialize", data={}))
        print("[UnityBridge] Scene initialized.")
        return resp

    def place_object(
        self,
        label: str,
        x: float,
        y: float,
        z: float,
        color: Optional[str] = None,
        scale: float = 1.0,
    ) -> Dict:
        """
        Place a labelled sphere in the Unity scene.

        Parameters
        ----------
        label  : Semantic label (e.g. "chair", "table").
        x      : Horizontal position.  Negative = left, Positive = right.
        y      : Vertical position.    0 = ground.  Set y = 0.5 to sit on the ground.
        z      : Depth position.       Negative = near camera, Positive = far.
        color  : Optional color name ("red", "blue", "green", "yellow",
                 "purple", "orange", "cyan", "pink") or hex string "#RRGGBB".
                 Auto-assigned from palette if omitted.
        scale  : Sphere scale multiplier (default 1.0 → diameter 1.0 unit).

        Returns
        -------
        dict with keys: status, id, label, position, color
        """
        payload: Dict[str, Any] = {
            "label": label,
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "scale": float(scale),
        }
        if color is not None:
            payload["color"] = color

        resp = self._check_error(self._request("place_object", data=payload))
        return resp

    def clear_scene(self) -> Dict:
        """Remove all placed objects from the Unity scene."""
        resp = self._check_error(self._request("clear_scene", data={}))
        return resp

    def get_scene_state(self) -> SceneState:
        """Return the current list of all placed objects."""
        resp = self._check_error(self._request("scene_state", method="GET"))
        objects = [
            PlacedObject(
                id=obj["id"],
                label=obj["label"],
                x=float(obj["x"]),
                y=float(obj["y"]),
                z=float(obj["z"]),
                color=obj.get("color", ""),
                scale=float(obj.get("scale", 1.0)),
            )
            for obj in resp.get("objects", [])
        ]
        return SceneState(objects=objects, count=resp.get("count", len(objects)))
