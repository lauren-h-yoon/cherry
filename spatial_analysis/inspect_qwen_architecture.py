#!/usr/bin/env python3
"""
inspect_qwen_architecture.py - Examine Qwen-VL model architecture for interpretability analysis.

This script loads the Qwen2.5-VL model and outputs detailed architecture information
including layer counts, attention heads, hidden dimensions, and hook points.
"""

import torch
import json
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig
from collections import OrderedDict


def inspect_model_architecture(model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """Load and inspect the Qwen-VL model architecture."""

    print("=" * 80)
    print(f"QWEN-VL ARCHITECTURE INSPECTION")
    print(f"Model: {model_name}")
    print("=" * 80)

    # Load config first (lightweight)
    print("\n[1/4] Loading model configuration...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    print("\n--- MODEL CONFIGURATION ---")
    config_dict = config.to_dict()

    # Key configuration values
    key_configs = [
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "vocab_size",
        "max_position_embeddings", "rope_theta", "vision_config"
    ]

    for key in key_configs:
        if key in config_dict:
            value = config_dict[key]
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

    # Load processor
    print("\n[2/4] Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Load model
    print("\n[3/4] Loading model (this may take a while)...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("\n[4/4] Analyzing architecture...")

    # Get model structure
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE TREE")
    print("=" * 80)

    def print_module_tree(module, prefix="", max_depth=3, current_depth=0):
        """Print module tree with limited depth."""
        if current_depth >= max_depth:
            return

        children = list(module.named_children())
        for i, (name, child) in enumerate(children):
            is_last = (i == len(children) - 1)
            connector = "└── " if is_last else "├── "

            # Get module info
            num_params = sum(p.numel() for p in child.parameters())
            param_str = f"({num_params:,} params)" if num_params > 0 else ""

            print(f"{prefix}{connector}{name}: {child.__class__.__name__} {param_str}")

            # Recurse
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_module_tree(child, new_prefix, max_depth, current_depth + 1)

    print_module_tree(model, max_depth=4)

    # Detailed layer analysis
    print("\n" + "=" * 80)
    print("DETAILED COMPONENT ANALYSIS")
    print("=" * 80)

    architecture_info = {
        "model_name": model_name,
        "components": {},
        "hook_points": [],
        "total_params": sum(p.numel() for p in model.parameters()),
    }

    # Vision encoder analysis
    print("\n--- VISION ENCODER ---")
    if hasattr(model, 'visual'):
        visual = model.visual
        print(f"Vision encoder type: {visual.__class__.__name__}")

        # Count vision layers
        vision_layers = []
        for name, module in visual.named_modules():
            if 'block' in name.lower() or 'layer' in name.lower():
                if name.count('.') <= 2:  # Only top-level blocks
                    vision_layers.append(name)

        print(f"Number of vision blocks: {len(set(vision_layers))}")

        # Vision config
        if hasattr(visual, 'config'):
            print(f"Vision hidden size: {getattr(visual.config, 'hidden_size', 'N/A')}")
            print(f"Vision num heads: {getattr(visual.config, 'num_attention_heads', 'N/A')}")

        architecture_info["components"]["vision_encoder"] = {
            "type": visual.__class__.__name__,
            "num_blocks": len(set(vision_layers)),
        }

        # Hook points for vision
        for name, module in visual.named_modules():
            if any(x in name for x in ['attn', 'attention', 'mlp', 'norm']):
                architecture_info["hook_points"].append(f"visual.{name}")

    # Language model analysis
    print("\n--- LANGUAGE MODEL ---")
    if hasattr(model, 'model'):
        lm = model.model
        print(f"LM type: {lm.__class__.__name__}")

        if hasattr(lm, 'layers'):
            num_layers = len(lm.layers)
            print(f"Number of LM layers: {num_layers}")

            # Analyze one layer
            if num_layers > 0:
                layer = lm.layers[0]
                print(f"\nLayer 0 structure:")
                for name, child in layer.named_children():
                    print(f"  - {name}: {child.__class__.__name__}")

                # Attention details
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    print(f"\nAttention configuration:")
                    print(f"  - num_heads: {getattr(attn, 'num_heads', 'N/A')}")
                    print(f"  - num_key_value_heads: {getattr(attn, 'num_key_value_heads', 'N/A')}")
                    print(f"  - head_dim: {getattr(attn, 'head_dim', 'N/A')}")

            architecture_info["components"]["language_model"] = {
                "type": lm.__class__.__name__,
                "num_layers": num_layers,
            }

            # Hook points for LM
            for i in range(min(num_layers, 3)):  # Sample first 3 layers
                architecture_info["hook_points"].extend([
                    f"model.layers.{i}.self_attn",
                    f"model.layers.{i}.self_attn.q_proj",
                    f"model.layers.{i}.self_attn.k_proj",
                    f"model.layers.{i}.self_attn.v_proj",
                    f"model.layers.{i}.self_attn.o_proj",
                    f"model.layers.{i}.mlp",
                    f"model.layers.{i}.input_layernorm",
                    f"model.layers.{i}.post_attention_layernorm",
                ])

    # Output head analysis
    print("\n--- OUTPUT HEAD ---")
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
        print(f"LM head type: {lm_head.__class__.__name__}")
        if hasattr(lm_head, 'weight'):
            print(f"Vocab size: {lm_head.weight.shape[0]}")
            print(f"Hidden size: {lm_head.weight.shape[1]}")

    # Summarize hook points
    print("\n" + "=" * 80)
    print("RECOMMENDED HOOK POINTS FOR ANALYSIS")
    print("=" * 80)

    print("\nVision Encoder:")
    print("  - visual.blocks.{i}.attn  (attention weights)")
    print("  - visual.blocks.{i}.mlp   (MLP activations)")
    print("  - visual.blocks.{i}       (block output)")

    print("\nLanguage Model:")
    print("  - model.layers.{i}.self_attn  (attention module)")
    print("  - model.layers.{i}.self_attn.o_proj  (attention output)")
    print("  - model.layers.{i}.mlp  (MLP activations)")
    print("  - model.layers.{i}  (layer output / residual stream)")

    print("\nCross-Modal:")
    print("  - model.embed_tokens  (token embeddings)")
    print("  - visual.merger  (vision-language projection)")

    # Memory usage
    print("\n" + "=" * 80)
    print("MEMORY ANALYSIS")
    print("=" * 80)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated memory (bf16): {total_params * 2 / 1e9:.2f} GB")
    print(f"Estimated memory (fp32): {total_params * 4 / 1e9:.2f} GB")

    architecture_info["total_params"] = total_params
    architecture_info["memory_bf16_gb"] = total_params * 2 / 1e9

    # Save architecture info to JSON
    output_path = "qwen_architecture_info.json"
    with open(output_path, "w") as f:
        json.dump(architecture_info, f, indent=2, default=str)
    print(f"\nArchitecture info saved to: {output_path}")

    # Print full module names for hooking
    print("\n" + "=" * 80)
    print("ALL MODULE NAMES (for hooking)")
    print("=" * 80)

    all_modules = []
    for name, module in model.named_modules():
        if name and '.' in name:
            depth = name.count('.')
            if depth <= 3:  # Limit depth
                all_modules.append(name)

    # Print grouped by top-level
    current_top = ""
    for name in sorted(all_modules)[:100]:  # Limit output
        top = name.split('.')[0]
        if top != current_top:
            print(f"\n[{top}]")
            current_top = top
        print(f"  {name}")

    if len(all_modules) > 100:
        print(f"\n  ... and {len(all_modules) - 100} more modules")

    return model, processor, architecture_info


if __name__ == "__main__":
    model, processor, arch_info = inspect_model_architecture()
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)
