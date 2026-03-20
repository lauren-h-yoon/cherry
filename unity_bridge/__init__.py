"""
unity_bridge — Python ↔ Unity HTTP interface for Cherry spatial evaluation.

Exposes:
  UnityBridge      : low-level HTTP client
  create_unity_tools : factory for LangChain-compatible tool objects
  PlacedObject / SceneState : data classes

Unity Coordinate System
-----------------------
  X  : Left (−) / Right (+)
  Y  : Ground (0) / Up (+) — floor at Y = 0, sphere sits at Y = 0.5
  Z  : Near (0) / Far (+)  — Z = 0 at camera, Z = 20 is far background

Scene bounds: X ∈ [−10, 10], Y ∈ [0, 10], Z ∈ [0, 20]
"""

from .bridge import UnityBridge, PlacedObject, SceneState
from .tools import create_unity_tools, create_zero_shot_tools

__all__ = ["UnityBridge", "PlacedObject", "SceneState", "create_unity_tools", "create_zero_shot_tools"]
