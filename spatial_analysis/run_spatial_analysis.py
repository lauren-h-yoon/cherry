#!/usr/bin/env python3
"""
run_spatial_analysis.py - Run activation and attention analysis on sampled cases.

This script:
1. Loads sampled cases from analysis_samples.json
2. Groups cases by image
3. Extracts activations at key layers
4. Extracts attention patterns
5. Compares success vs failure patterns
6. Saves results for visualization

Usage:
    python run_spatial_analysis.py --samples analysis_samples.json --output spatial_analysis_results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


def load_qwen_model():
    """Load Qwen2.5-VL-7B model."""
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Need eager for attention weights
    )
    model.eval()

    print("Model loaded successfully")
    return model, processor


def find_image_for_query(image_id: str) -> tuple:
    """Find the image path and graph path for an image ID."""
    # Look for graph file
    graph_files = list(Path("spatial_outputs_coco_50").glob(f"{image_id}_spatial_graph.json"))
    if not graph_files:
        return None, None

    graph_path = graph_files[0]

    # Get image path from graph
    with open(graph_path) as f:
        graph_data = json.load(f)

    raw_image_path = graph_data.get("image_path", "")

    # Try to find the image
    candidates = [
        Path(raw_image_path),
        Path("val2017") / f"{image_id}.jpg",
        Path(graph_path).parent / raw_image_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate), str(graph_path)

    return None, str(graph_path)


def extract_activations_and_attention(
    model,
    processor,
    image_path: str,
    prompt: str,
    hook_points: list,
):
    """Extract activations and attention for a single query."""
    from qwen_vl_utils import process_vision_info

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
    inputs = inputs.to(model.device)

    # Storage for activations
    activations = {}
    attention_weights = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach().cpu()
                if len(output) > 1 and output[1] is not None:
                    attention_weights[name] = output[1].detach().cpu()
            else:
                activations[name] = output.detach().cpu()
        return hook_fn

    # Register hooks
    for name, module in model.named_modules():
        if name in hook_points:
            handle = module.register_forward_hook(make_hook(name))
            hooks.append(handle)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return {
        "activations": activations,
        "attention_weights": attention_weights,
        "input_ids": inputs.input_ids.cpu(),
        "logits": outputs.logits[:, -1, :].cpu() if hasattr(outputs, 'logits') else None,
    }


def compute_activation_stats(activations: dict) -> dict:
    """Compute statistics on activations for comparison."""
    stats = {}
    for name, act in activations.items():
        if act is None:
            continue
        act_np = act.float().numpy()

        # Pool across sequence length if needed
        if act_np.ndim == 3:
            # (batch, seq, hidden) -> mean pool
            act_pooled = act_np.mean(axis=1)[0]  # (hidden,)
        elif act_np.ndim == 2:
            act_pooled = act_np[0]
        else:
            act_pooled = act_np.flatten()

        stats[name] = {
            "mean": float(np.mean(act_pooled)),
            "std": float(np.std(act_pooled)),
            "norm": float(np.linalg.norm(act_pooled)),
            "max": float(np.max(np.abs(act_pooled))),
            "shape": list(act.shape),
        }

    return stats


def analyze_attention_focus(attention_weights: dict, num_image_tokens: int = 256) -> dict:
    """Analyze where attention is focused (image vs text)."""
    focus_stats = {}
    for name, attn in attention_weights.items():
        if attn is None:
            continue

        # attn shape: (batch, heads, q_len, kv_len)
        attn_np = attn.float().numpy()

        if attn_np.ndim == 4:
            # Average over heads, take last query position
            attn_last = attn_np[0, :, -1, :].mean(axis=0)  # (kv_len,)
        else:
            attn_last = attn_np.flatten()

        # Split into image and text attention
        if len(attn_last) > num_image_tokens:
            image_attn = attn_last[:num_image_tokens].sum()
            text_attn = attn_last[num_image_tokens:].sum()
        else:
            image_attn = attn_last.sum()
            text_attn = 0

        total = image_attn + text_attn + 1e-8
        focus_stats[name] = {
            "image_attention_ratio": float(image_attn / total),
            "text_attention_ratio": float(text_attn / total),
            "attention_entropy": float(-np.sum(attn_last * np.log(attn_last + 1e-8))),
        }

    return focus_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", "-s", default="spatial_analysis/analysis_samples.json")
    parser.add_argument("--output", "-o", default="spatial_analysis_results")
    parser.add_argument("--max-cases", type=int, default=None, help="Max cases to process (for testing)")
    args = parser.parse_args()

    # Load samples
    with open(args.samples) as f:
        samples_data = json.load(f)

    cases = samples_data["cases"]
    if args.max_cases:
        cases = cases[:args.max_cases]

    print(f"Processing {len(cases)} sampled cases")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, processor = load_qwen_model()

    # Define hook points (key layers for analysis)
    hook_points = [
        # Vision encoder layers
        "model.visual.blocks.7",
        "model.visual.blocks.23",
        "model.visual.blocks.31",
        # Merger
        "model.visual.merger",
        # LM layers (correct path for Qwen2.5-VL)
        "model.language_model.layers.0",
        "model.language_model.layers.7",
        "model.language_model.layers.14",
        "model.language_model.layers.21",
        "model.language_model.layers.27",
    ]

    # Process cases
    results = []
    success_activations = []
    failure_activations = []

    for idx, case in enumerate(cases):
        query_id = case["query_id"]
        image_id = case["image_id"]
        print(f"\n[{idx+1}/{len(cases)}] Processing {query_id} (image: {image_id})")

        # Find image
        image_path, graph_path = find_image_for_query(image_id)
        if not image_path or not Path(image_path).exists():
            print(f"  Warning: Image not found for {image_id}")
            continue

        print(f"  Image: {image_path}")
        print(f"  Prompt: {case['prompt'][:60]}...")
        print(f"  Correct: {case['correct']}")

        try:
            # Extract activations and attention
            extracted = extract_activations_and_attention(
                model, processor,
                image_path,
                case["prompt"],
                hook_points
            )

            # Compute statistics
            act_stats = compute_activation_stats(extracted["activations"])
            attn_focus = analyze_attention_focus(extracted["attention_weights"])

            result = {
                "query_id": query_id,
                "task_type": case["task_type"],
                "frame_type": case["frame_type"],
                "relation_axis": case["relation_axis"],
                "correct": case["correct"],
                "ground_truth": case["ground_truth"],
                "model_answer": case["model_answer"],
                "activation_stats": act_stats,
                "attention_focus": attn_focus,
            }
            results.append(result)

            # Group by correctness for comparison
            if case["correct"]:
                success_activations.append(act_stats)
            else:
                failure_activations.append(act_stats)

            # Save individual result
            case_dir = output_dir / query_id
            case_dir.mkdir(exist_ok=True)
            with open(case_dir / "analysis.json", "w") as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate analysis
    print("\n" + "=" * 60)
    print("AGGREGATE ANALYSIS")
    print("=" * 60)

    # Compare success vs failure activation norms
    if success_activations and failure_activations:
        print("\nActivation Norm Comparison (Success vs Failure):")

        for layer in hook_points:
            success_norms = [s.get(layer, {}).get("norm", 0) for s in success_activations]
            failure_norms = [s.get(layer, {}).get("norm", 0) for s in failure_activations]

            if success_norms and failure_norms:
                success_mean = np.mean([n for n in success_norms if n > 0])
                failure_mean = np.mean([n for n in failure_norms if n > 0])

                layer_short = layer.split(".")[-1] if "layers" in layer else layer.split(".")[-2:]
                print(f"  {layer_short}: Success={success_mean:.2f}, Failure={failure_mean:.2f}")

    # Save aggregate results
    aggregate = {
        "num_cases": len(results),
        "num_success": len(success_activations),
        "num_failure": len(failure_activations),
        "results": results,
    }

    with open(output_dir / "aggregate_analysis.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Aggregate analysis: {output_dir / 'aggregate_analysis.json'}")


if __name__ == "__main__":
    main()
