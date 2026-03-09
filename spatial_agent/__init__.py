"""
spatial_agent - Agentic Spatial Reasoning System

An evaluation framework for testing VLM/agent spatial reasoning capabilities
using depth-extracted waypoints from the Cherry pipeline.

Supports:
- Egocentric navigation (z-order based)
- Allocentric reasoning (object-to-object relationships)
- Q&A evaluation (spatial relationship questions)
- Perspective-shift tasks (view from object positions)
- Embodied spatial intelligence evaluation (primitive tools as output mechanisms)
"""

from .annotator import SpatialAnnotator, SceneConfig, Waypoint
from .state import AgentState, WaypointView
from .viewpoint_annotator import (
    ViewpointAnnotator,
    ViewpointMarker,
    annotate_for_egocentric_task,
    annotate_for_allocentric_path,
    annotate_for_allocentric_qa,
    get_standard_viewpoints
)

# Embodied spatial intelligence components
from .primitive_tools import (
    RecordedOutputs,
    EmbodiedState,
    DrawPathTool,
    PointToTool,
    MarkRegionTool,
    MoveToTool,
    RotateTool,
    LookAtTool,
    create_primitive_tools
)
from .ground_truth import GroundTruth, ObjectInfo
from .evaluator import (
    SpatialEvaluator,
    EvaluationReport,
    PointingResult,
    PathResult,
    RegionResult
)

# Model provider abstraction
from .model_providers import (
    VLMProvider,
    ClaudeProvider,
    QwenVLProvider,
    VLLMServerProvider,
    OpenAIProvider,
    OllamaProvider,
    create_model_provider,
    list_available_providers,
    ModelResponse,
    ToolCall
)
from .tasks import (
    SpatialTask,
    TaskType,
    Difficulty,
    TaskGenerator
)

# Lazy imports for LangChain components (may not be installed)
def get_tools():
    from .tools import create_tools, GetWaypointsTool, MoveToTool, RotateTool, ScaleTool
    return create_tools, GetWaypointsTool, MoveToTool, RotateTool, ScaleTool

def get_agent():
    from .agent import SpatialReasoningAgent
    return SpatialReasoningAgent

def get_allocentric_agent():
    from .agent import AllocentricReasoningAgent
    return AllocentricReasoningAgent

def get_allocentric():
    from .allocentric import AllocentricRelationships, SpatialRelation
    return AllocentricRelationships, SpatialRelation

def get_allocentric_tools():
    from .allocentric_tools import (
        create_allocentric_tools,
        GetSpatialRelationTool,
        QuerySceneTool,
        PerspectiveShiftTool,
        ResolveAllocentricGoalTool
    )
    return (
        create_allocentric_tools,
        GetSpatialRelationTool,
        QuerySceneTool,
        PerspectiveShiftTool,
        ResolveAllocentricGoalTool
    )

def get_evaluator():
    from .allocentric_eval import (
        AllocentricEvaluator,
        AllocentricBenchmark,
        AllocentricTask,
        EvaluationResult
    )
    return (
        AllocentricEvaluator,
        AllocentricBenchmark,
        AllocentricTask,
        EvaluationResult
    )

def get_embodied_agent():
    from .embodied_agent import EmbodiedSpatialAgent, EmbodiedEvaluationRunner
    return EmbodiedSpatialAgent, EmbodiedEvaluationRunner

def get_query_integration():
    from .query_integration import (
        QueryLoader,
        AllocentricQuery,
        AllocentricEvaluator,
        load_and_convert_queries
    )
    return QueryLoader, AllocentricQuery, AllocentricEvaluator, load_and_convert_queries

__all__ = [
    # Core components
    'SpatialAnnotator',
    'SceneConfig',
    'Waypoint',
    'AgentState',
    'WaypointView',
    # Viewpoint annotation
    'ViewpointAnnotator',
    'ViewpointMarker',
    'annotate_for_egocentric_task',
    'annotate_for_allocentric_path',
    'annotate_for_allocentric_qa',
    'get_standard_viewpoints',
    # Lazy loaders
    'get_tools',
    'get_agent',
    'get_allocentric_agent',
    'get_allocentric',
    'get_allocentric_tools',
    'get_evaluator',
    # Embodied spatial intelligence
    'RecordedOutputs',
    'EmbodiedState',
    'DrawPathTool',
    'PointToTool',
    'MarkRegionTool',
    'MoveToTool',
    'RotateTool',
    'LookAtTool',
    'create_primitive_tools',
    'GroundTruth',
    'ObjectInfo',
    'SpatialEvaluator',
    'EvaluationReport',
    'PointingResult',
    'PathResult',
    'RegionResult',
    'SpatialTask',
    'TaskType',
    'Difficulty',
    'TaskGenerator',
    'get_embodied_agent',
    'get_query_integration',
    # Model providers
    'VLMProvider',
    'ClaudeProvider',
    'QwenVLProvider',
    'VLLMServerProvider',
    'OpenAIProvider',
    'OllamaProvider',
    'create_model_provider',
    'list_available_providers',
    'ModelResponse',
    'ToolCall',
]
