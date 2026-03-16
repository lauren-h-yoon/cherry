#!/usr/bin/env python3
"""
activation_extractor.py - Extract activations from Qwen-VL for interpretability analysis.

This module provides hook-based activation extraction from specific layers of
Qwen2.5-VL for probing and analysis of spatial reasoning.

Key hook points:
- Vision encoder blocks (model.visual.blocks.{i})
- Vision-language merger (model.visual.merger)
- Language model layers (model.language_model.layers.{i})
- Attention outputs (model.language_model.layers.{i}.self_attn)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ActivationStore:
    """Container for storing extracted activations."""
    activations: Dict[str, torch.Tensor] = field(default_factory=dict)
    attention_weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def clear(self):
        self.activations.clear()
        self.attention_weights.clear()

    def save(self, path: str):
        """Save activations to disk."""
        save_dict = {
            "activations": {k: v.cpu().numpy().tolist() for k, v in self.activations.items()},
            "attention_weights": {k: v.cpu().numpy().tolist() for k, v in self.attention_weights.items()},
            "metadata": self.metadata
        }
        with open(path, "w") as f:
            json.dump(save_dict, f)


class ActivationExtractor:
    """
    Hook-based activation extractor for Qwen2.5-VL.

    Usage:
        extractor = ActivationExtractor(model)
        extractor.register_hooks(["model.visual.blocks.31", "model.language_model.layers.14"])

        # Forward pass
        outputs = model.generate(**inputs)

        # Get activations
        activations = extractor.get_activations()
        extractor.clear()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.store = ActivationStore()
        self._module_cache = {}
        self._build_module_cache()

    def _build_module_cache(self):
        """Build cache of named modules for quick lookup."""
        for name, module in self.model.named_modules():
            self._module_cache[name] = module

    def _get_module(self, name: str) -> Optional[nn.Module]:
        """Get module by name."""
        return self._module_cache.get(name)

    def register_hooks(
        self,
        layer_names: List[str],
        capture_attention: bool = False
    ):
        """
        Register forward hooks on specified layers.

        Args:
            layer_names: List of module names to hook
            capture_attention: Whether to capture attention weights
        """
        self.remove_hooks()

        for layer_name in layer_names:
            module = self._get_module(layer_name)
            if module is None:
                print(f"Warning: Module '{layer_name}' not found")
                continue

            # Create closure to capture layer_name
            def make_hook(name):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        # For attention modules, output is (hidden_states, attention_weights, ...)
                        self.store.activations[name] = output[0].detach()
                        if capture_attention and len(output) > 1 and output[1] is not None:
                            self.store.attention_weights[name] = output[1].detach()
                    else:
                        self.store.activations[name] = output.detach()
                return hook_fn

            handle = module.register_forward_hook(make_hook(layer_name))
            self.hooks.append(handle)

        print(f"Registered {len(self.hooks)} hooks")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear(self):
        """Clear stored activations."""
        self.store.clear()

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get current stored activations."""
        return self.store.activations

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get current stored attention weights."""
        return self.store.attention_weights

    def __del__(self):
        self.remove_hooks()


class SpatialProbe(nn.Module):
    """
    Linear probe for spatial relation classification.

    Used to test what spatial information is encoded at different layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        pooling: str = "mean"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pooling = pooling

        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Activations of shape (batch, seq_len, hidden_dim) or (batch, hidden_dim)

        Returns:
            logits of shape (batch, num_classes)
        """
        if x.dim() == 3:
            if self.pooling == "mean":
                x = x.mean(dim=1)
            elif self.pooling == "first":
                x = x[:, 0]
            elif self.pooling == "last":
                x = x[:, -1]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.classifier(x)


def get_recommended_hook_points(model_type: str = "qwen2.5-vl-7b") -> Dict[str, List[str]]:
    """
    Get recommended hook points for interpretability analysis.

    Returns dict with categories:
    - vision_early: Early vision blocks (low-level features)
    - vision_mid: Middle vision blocks (spatial encoding)
    - vision_late: Late vision blocks (high-level features)
    - merger: Vision-language projection
    - lm_early: Early LM layers (query understanding)
    - lm_mid: Middle LM layers (reasoning)
    - lm_late: Late LM layers (decision)
    """
    if "qwen2.5-vl-7b" in model_type.lower():
        return {
            "vision_early": [
                "model.visual.blocks.0",
                "model.visual.blocks.3",
                "model.visual.blocks.7",  # full attention
            ],
            "vision_mid": [
                "model.visual.blocks.15",  # full attention
                "model.visual.blocks.19",
            ],
            "vision_late": [
                "model.visual.blocks.23",  # full attention
                "model.visual.blocks.27",
                "model.visual.blocks.31",  # full attention
            ],
            "merger": [
                "model.visual.merger",
                "model.visual.merger.mlp",
            ],
            "lm_early": [
                "model.language_model.layers.0",
                "model.language_model.layers.3",
                "model.language_model.layers.7",
            ],
            "lm_mid": [
                "model.language_model.layers.10",
                "model.language_model.layers.14",
                "model.language_model.layers.17",
            ],
            "lm_late": [
                "model.language_model.layers.21",
                "model.language_model.layers.24",
                "model.language_model.layers.27",
            ],
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def extract_activations_for_query(
    model,
    processor,
    image_path: str,
    prompt: str,
    hook_points: List[str],
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Extract activations for a single query.

    Args:
        model: Qwen-VL model
        processor: Qwen-VL processor
        image_path: Path to image
        prompt: Query prompt
        hook_points: List of module names to extract activations from
        device: Device to use

    Returns:
        Dict mapping hook point names to activation tensors
    """
    from qwen_vl_utils import process_vision_info

    extractor = ActivationExtractor(model)
    extractor.register_hooks(hook_points, capture_attention=True)

    # Build messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    # Forward pass (just encode, don't generate)
    with torch.no_grad():
        # Get the model's outputs without generation to capture activations
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    activations = extractor.get_activations()
    attention_weights = extractor.get_attention_weights()

    extractor.remove_hooks()

    return {
        "activations": activations,
        "attention_weights": attention_weights,
        "input_ids": inputs.input_ids,
    }


def batch_extract_activations(
    model,
    processor,
    image_path: str,
    queries: List[Dict],
    hook_points: List[str],
    output_dir: str,
    device: str = "cuda"
) -> List[Dict]:
    """
    Extract activations for multiple queries and save to disk.

    Args:
        model: Qwen-VL model
        processor: Qwen-VL processor
        image_path: Path to benchmark image
        queries: List of query dicts with 'prompt', 'ground_truth_answer', etc.
        hook_points: List of module names to extract activations from
        output_dir: Directory to save activations
        device: Device to use

    Returns:
        List of activation metadata dicts
    """
    import numpy as np

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, query in enumerate(queries):
        print(f"[{idx+1}/{len(queries)}] Extracting: {query['query_id']}")

        extracted = extract_activations_for_query(
            model, processor,
            image_path,
            query["prompt"],
            hook_points,
            device
        )

        # Save activations
        query_dir = output_path / query["query_id"]
        query_dir.mkdir(exist_ok=True)

        for name, tensor in extracted["activations"].items():
            # Save as numpy for efficiency
            np.save(query_dir / f"{name.replace('.', '_')}_act.npy", tensor.cpu().numpy())

        for name, tensor in extracted["attention_weights"].items():
            np.save(query_dir / f"{name.replace('.', '_')}_attn.npy", tensor.cpu().numpy())

        # Save metadata
        metadata = {
            "query_id": query["query_id"],
            "task_type": query["task_type"],
            "frame_type": query["frame_type"],
            "relation_axis": query.get("relation_axis"),
            "ground_truth": query["ground_truth_answer"],
            "prompt": query["prompt"],
            "hook_points": hook_points,
        }
        with open(query_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        results.append(metadata)

    # Save index
    with open(output_path / "index.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved activations for {len(results)} queries to {output_path}")
    return results


if __name__ == "__main__":
    # Test the module
    print("Recommended hook points for Qwen2.5-VL-7B:")
    for category, points in get_recommended_hook_points().items():
        print(f"\n{category}:")
        for point in points:
            print(f"  - {point}")
