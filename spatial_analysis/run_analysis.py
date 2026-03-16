#!/usr/bin/env python3
"""
run_analysis.py - Main script for running full interpretability analysis on Qwen-VL.

This script orchestrates:
1. Baseline evaluation on test images
2. Activation extraction for selected cases
3. Attention visualization
4. Success vs failure comparison

Usage:
    python run_analysis.py --graph <spatial_graph.json> --output-dir <output>
    python run_analysis.py --graph <spatial_graph.json> --phase eval --output-dir <output>
    python run_analysis.py --graph <spatial_graph.json> --phase attention --output-dir <output>
"""

import argparse
import json
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_and_processor():
    """Load Qwen2.5-VL-7B model and processor."""
    from transformers import AutoProcessor, AutoModelForVision2Seq

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Model loaded successfully")
    return model, processor


def phase_eval(args):
    """Phase 1: Run baseline evaluation."""
    from run_qwen_eval import evaluate_queries, load_qwen_model, generate_response, QUERY_SYSTEM_PROMPT, normalize_answer
    from query_benchmark.generator import generate_queries
    from query_benchmark.ground_truth import load_graph, resolve_stage1_labeled_image, resolve_image_path

    output_dir = Path(args.output_dir) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE EVALUATION")
    print("=" * 60)

    # Generate queries
    generated = generate_queries(
        args.graph,
        include_object_retrieval=False,
        queries_per_bucket=args.queries_per_bucket,
        seed=args.seed,
    )
    queries = generated["queries"]
    print(f"Generated {len(queries)} queries")

    # Save queries
    with open(output_dir / "queries.json", "w") as f:
        json.dump(generated, f, indent=2)

    # Get benchmark image
    graph_data = load_graph(args.graph)
    existing_labeled = resolve_stage1_labeled_image(args.graph)
    benchmark_image = str(existing_labeled) if existing_labeled else str(resolve_image_path(graph_data, args.graph))
    print(f"Benchmark image: {benchmark_image}")

    # Load model and evaluate
    model, processor = load_qwen_model()
    evaluation = evaluate_queries(model, processor, benchmark_image, queries, output_dir / "evaluation_results.json")

    # Save final results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(evaluation, f, indent=2)

    print(f"\nOverall: {evaluation['overall']['accuracy']:.1%}")
    return evaluation


def phase_extract_activations(args, eval_results=None):
    """Phase 2: Extract activations for selected cases."""
    from activation_extractor import (
        ActivationExtractor,
        batch_extract_activations,
        get_recommended_hook_points
    )
    from query_benchmark.ground_truth import load_graph, resolve_stage1_labeled_image, resolve_image_path

    output_dir = Path(args.output_dir) / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PHASE 2: ACTIVATION EXTRACTION")
    print("=" * 60)

    # Load evaluation results if not provided
    if eval_results is None:
        eval_path = Path(args.output_dir) / "eval" / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path) as f:
                eval_results = json.load(f)
        else:
            print("Error: No evaluation results found. Run --phase eval first.")
            return None

    # Select cases
    results = eval_results.get("results", [])
    success_cases = [r for r in results if r["correct"]]
    failure_cases = [r for r in results if not r["correct"]]

    print(f"Success cases: {len(success_cases)}")
    print(f"Failure cases: {len(failure_cases)}")

    # Select up to 5 of each
    selected_success = success_cases[:min(5, len(success_cases))]
    selected_failure = failure_cases[:min(5, len(failure_cases))]
    selected_queries = selected_success + selected_failure

    # Save selection
    selection = {
        "success": [r["query_id"] for r in selected_success],
        "failure": [r["query_id"] for r in selected_failure],
        "details": {r["query_id"]: r for r in selected_queries}
    }
    with open(output_dir / "selected_cases.json", "w") as f:
        json.dump(selection, f, indent=2)

    # Get hook points
    hook_points = []
    for category, points in get_recommended_hook_points().items():
        hook_points.extend(points[:2])  # Top 2 from each category

    print(f"Hook points: {len(hook_points)}")
    print(f"Selected queries: {len(selected_queries)}")

    # Get benchmark image
    graph_data = load_graph(args.graph)
    existing_labeled = resolve_stage1_labeled_image(args.graph)
    benchmark_image = str(existing_labeled) if existing_labeled else str(resolve_image_path(graph_data, args.graph))

    # Load model
    model, processor = load_model_and_processor()

    # Extract activations
    batch_extract_activations(
        model, processor,
        benchmark_image,
        selected_queries,
        hook_points,
        str(output_dir),
        device=model.device
    )

    return selection


def phase_attention(args, eval_results=None):
    """Phase 3: Attention visualization."""
    from attention_visualizer import (
        visualize_attention_on_image,
        visualize_layer_attention_progression,
        compare_success_failure_attention
    )
    from activation_extractor import ActivationExtractor, extract_activations_for_query
    from query_benchmark.ground_truth import load_graph, resolve_stage1_labeled_image, resolve_image_path

    output_dir = Path(args.output_dir) / "attention"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PHASE 3: ATTENTION VISUALIZATION")
    print("=" * 60)

    # Load evaluation results
    if eval_results is None:
        eval_path = Path(args.output_dir) / "eval" / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path) as f:
                eval_results = json.load(f)
        else:
            print("Error: No evaluation results found. Run --phase eval first.")
            return

    # Get benchmark image
    graph_data = load_graph(args.graph)
    existing_labeled = resolve_stage1_labeled_image(args.graph)
    benchmark_image = str(existing_labeled) if existing_labeled else str(resolve_image_path(graph_data, args.graph))

    # Load model
    model, processor = load_model_and_processor()

    # Layers to visualize
    attn_layers = [
        "model.language_model.layers.0.self_attn",
        "model.language_model.layers.7.self_attn",
        "model.language_model.layers.14.self_attn",
        "model.language_model.layers.21.self_attn",
        "model.language_model.layers.27.self_attn",
    ]

    results = eval_results.get("results", [])
    success_cases = [r for r in results if r["correct"]][:3]
    failure_cases = [r for r in results if not r["correct"]][:3]

    success_attentions = []
    failure_attentions = []

    # Extract attention for success cases
    print("\nExtracting attention for success cases...")
    for case in success_cases:
        print(f"  - {case['query_id']}")
        extracted = extract_activations_for_query(
            model, processor,
            benchmark_image,
            case["prompt"],
            attn_layers,
            device=model.device
        )
        success_attentions.append(extracted["attention_weights"])

        # Individual visualization
        if extracted["attention_weights"]:
            layer_name = list(extracted["attention_weights"].keys())[0]
            visualize_attention_on_image(
                benchmark_image,
                extracted["attention_weights"][layer_name],
                str(output_dir / f"success_{case['query_id']}_attn.png"),
                title=f"Success: {case['task_type']} - {case['relation_axis']}"
            )

    # Extract attention for failure cases
    print("\nExtracting attention for failure cases...")
    for case in failure_cases:
        print(f"  - {case['query_id']}")
        extracted = extract_activations_for_query(
            model, processor,
            benchmark_image,
            case["prompt"],
            attn_layers,
            device=model.device
        )
        failure_attentions.append(extracted["attention_weights"])

        # Individual visualization
        if extracted["attention_weights"]:
            layer_name = list(extracted["attention_weights"].keys())[0]
            visualize_attention_on_image(
                benchmark_image,
                extracted["attention_weights"][layer_name],
                str(output_dir / f"failure_{case['query_id']}_attn.png"),
                title=f"Failure: {case['task_type']} - {case['relation_axis']}"
            )

    # Comparison visualization
    if success_attentions and failure_attentions:
        for layer in attn_layers:
            safe_layer = layer.replace(".", "_")
            compare_success_failure_attention(
                benchmark_image,
                success_attentions,
                failure_attentions,
                layer,
                str(output_dir / f"compare_{safe_layer}.png"),
                success_queries=[c["prompt"][:50] for c in success_cases],
                failure_queries=[c["prompt"][:50] for c in failure_cases]
            )

    print(f"\nAttention visualizations saved to {output_dir}")


def phase_summary(args):
    """Generate summary report."""
    output_dir = Path(args.output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    # Load eval results
    eval_path = output_dir / "eval" / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)

        print(f"\n[EVALUATION RESULTS]")
        print(f"Model: {eval_results['model']['model_name']}")
        print(f"Overall: {eval_results['overall']['accuracy']:.1%} ({eval_results['overall']['correct']}/{eval_results['overall']['count']})")

        print(f"\nBy Task Type:")
        for task, stats in eval_results.get("by_task_type", {}).items():
            print(f"  {task}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['count']})")

        # Breakdown by relation axis
        print(f"\nBy Relation Axis:")
        by_axis = {}
        for r in eval_results.get("results", []):
            axis = r.get("relation_axis", "unknown")
            if axis not in by_axis:
                by_axis[axis] = {"correct": 0, "total": 0}
            by_axis[axis]["total"] += 1
            by_axis[axis]["correct"] += int(r["correct"])

        for axis, stats in sorted(by_axis.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] else 0
            print(f"  {axis}: {acc:.1%} ({stats['correct']}/{stats['total']})")

    # Check for activation files
    act_dir = output_dir / "activations"
    if act_dir.exists():
        index_path = act_dir / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            print(f"\n[ACTIVATIONS]")
            print(f"Extracted for {len(index)} queries")

    # Check for attention visualizations
    attn_dir = output_dir / "attention"
    if attn_dir.exists():
        attn_files = list(attn_dir.glob("*.png"))
        print(f"\n[ATTENTION VISUALIZATIONS]")
        print(f"Generated {len(attn_files)} visualization files")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run interpretability analysis on Qwen-VL")
    parser.add_argument("--graph", "-g", required=True, help="Path to spatial_graph.json")
    parser.add_argument("--output-dir", "-o", default="analysis_outputs", help="Output directory")
    parser.add_argument("--phase", choices=["all", "eval", "activations", "attention", "summary"],
                        default="all", help="Which phase to run")
    parser.add_argument("--queries-per-bucket", type=int, default=4, help="Queries per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_results = None

    if args.phase in ["all", "eval"]:
        eval_results = phase_eval(args)

    if args.phase in ["all", "activations"]:
        phase_extract_activations(args, eval_results)

    if args.phase in ["all", "attention"]:
        phase_attention(args, eval_results)

    if args.phase in ["all", "summary"]:
        phase_summary(args)


if __name__ == "__main__":
    main()
