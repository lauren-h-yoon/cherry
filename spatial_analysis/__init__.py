"""
spatial_analysis - Interpretability analysis tools for VLM spatial reasoning.

Modules:
- activation_extractor: Hook-based activation extraction from Qwen-VL
- attention_visualizer: Attention pattern visualization
- run_qwen_eval: Qwen-VL evaluation script
- run_analysis: Main analysis orchestration
"""

from .activation_extractor import (
    ActivationExtractor,
    ActivationStore,
    SpatialProbe,
    get_recommended_hook_points,
    extract_activations_for_query,
    batch_extract_activations,
)

from .attention_visualizer import (
    attention_to_heatmap,
    visualize_attention_on_image,
    visualize_layer_attention_progression,
    compare_success_failure_attention,
)

__all__ = [
    "ActivationExtractor",
    "ActivationStore",
    "SpatialProbe",
    "get_recommended_hook_points",
    "extract_activations_for_query",
    "batch_extract_activations",
    "attention_to_heatmap",
    "visualize_attention_on_image",
    "visualize_layer_attention_progression",
    "compare_success_failure_attention",
]
