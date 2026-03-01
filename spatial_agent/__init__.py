"""
spatial_agent - Agentic Spatial Reasoning System

An evaluation framework for testing VLM/agent spatial reasoning capabilities
using depth-extracted waypoints from the Cherry pipeline.
"""

from .annotator import SpatialAnnotator, SceneConfig, Waypoint
from .state import AgentState, WaypointView

# Lazy imports for LangChain components (may not be installed)
def get_tools():
    from .tools import create_tools, GetWaypointsTool, MoveToTool, RotateTool, ScaleTool
    return create_tools, GetWaypointsTool, MoveToTool, RotateTool, ScaleTool

def get_agent():
    from .agent import SpatialReasoningAgent
    return SpatialReasoningAgent

__all__ = [
    'SpatialAnnotator',
    'SceneConfig',
    'Waypoint',
    'AgentState',
    'WaypointView',
    'get_tools',
    'get_agent',
]
