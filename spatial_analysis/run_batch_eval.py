#!/usr/bin/env python3
"""
run_batch_eval.py - Run Qwen-VL evaluation on multiple images.

This script evaluates Qwen2.5-VL-7B on all spatial graphs in a directory
and aggregates results across images.

Usage:
    python run_batch_eval.py --input-dir spatial_outputs_coco_50 --output-dir qwen_batch_eval
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from query_benchmark.generator import generate_queries
from query_benchmark.ground_truth import (
    load_graph,
    resolve_image_path,
    resolve_stage1_labeled_image,
)


QUERY_SYSTEM_PROMPT = """You are answering spatial reasoning questions about an image with labeled bounding boxes.

Use the labels shown in the image exactly.
Return only the answer.
Do not explain your reasoning.
If the valid answer choices are shown in the prompt, output exactly one of them and nothing else.
"""


def _basic_normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().replace("_", " ").strip(".").split())


def normalize_answer(text: str, candidate_answers=None) -> str:
    normalized = _basic_normalize(text)
    if not candidate_answers:
        return normalized.replace(" ", "_")

    candidate_map = {_basic_normalize(answer): answer.lower() for answer in candidate_answers}
    if normalized in candidate_map:
        return candidate_map[normalized]

    for normalized_candidate, original_candidate in sorted(
        candidate_map.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if normalized_candidate and normalized_candidate in normalized:
            return original_candidate

    return normalized.replace(" ", "_")


def load_qwen_model():
    """Load Qwen2.5-VL-7B model."""
    import torch
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


def generate_response(model, processor, image_path: str, prompt: str, system_prompt: str = "") -> str:
    """Generate a response from Qwen."""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]
    })

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

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()


def evaluate_single_image(model, processor, graph_path: str, queries_per_bucket: int = 4, seed: int = 42):
    """Evaluate on a single image."""
    # Generate queries
    generated = generate_queries(
        graph_path,
        include_object_retrieval=False,
        queries_per_bucket=queries_per_bucket,
        seed=seed,
    )
    queries = generated["queries"]

    # Get benchmark image
    graph_data = load_graph(graph_path)
    existing_labeled = resolve_stage1_labeled_image(graph_path)
    benchmark_image = str(existing_labeled) if existing_labeled else str(resolve_image_path(graph_data, graph_path))

    results = []
    correct = 0

    for query in queries:
        try:
            response = generate_response(
                model, processor,
                benchmark_image,
                query["prompt"],
                QUERY_SYSTEM_PROMPT
            )
        except Exception as e:
            print(f"    Error: {e}")
            response = "ERROR"

        model_answer = normalize_answer(response, query.get("candidate_answers"))
        gt = normalize_answer(query["ground_truth_answer"], query.get("candidate_answers"))
        is_correct = model_answer == gt

        if is_correct:
            correct += 1

        results.append({
            "query_id": query["query_id"],
            "task_type": query["task_type"],
            "frame_type": query["frame_type"],
            "query_subtype": query["query_subtype"],
            "relation_axis": query.get("relation_axis"),
            "prompt": query["prompt"],
            "ground_truth_answer": query["ground_truth_answer"],
            "model_answer": response,
            "normalized_model_answer": model_answer,
            "correct": is_correct,
        })

    return {
        "image": Path(graph_path).stem.replace("_spatial_graph", ""),
        "num_queries": len(queries),
        "correct": correct,
        "accuracy": correct / len(queries) if queries else 0,
        "results": results,
    }


def aggregate_results(all_results: list) -> dict:
    """Aggregate results across all images."""
    total_queries = 0
    total_correct = 0
    all_query_results = []

    # Collect all results
    for img_result in all_results:
        total_queries += img_result["num_queries"]
        total_correct += img_result["correct"]
        all_query_results.extend(img_result["results"])

    # Breakdown by task type
    by_task_type = {}
    for r in all_query_results:
        task = r["task_type"]
        if task not in by_task_type:
            by_task_type[task] = {"count": 0, "correct": 0}
        by_task_type[task]["count"] += 1
        by_task_type[task]["correct"] += int(r["correct"])
    for task in by_task_type:
        by_task_type[task]["accuracy"] = by_task_type[task]["correct"] / by_task_type[task]["count"]

    # Breakdown by relation axis
    by_axis = {}
    for r in all_query_results:
        axis = r.get("relation_axis", "unknown")
        if axis not in by_axis:
            by_axis[axis] = {"count": 0, "correct": 0}
        by_axis[axis]["count"] += 1
        by_axis[axis]["correct"] += int(r["correct"])
    for axis in by_axis:
        by_axis[axis]["accuracy"] = by_axis[axis]["correct"] / by_axis[axis]["count"]

    # Breakdown by frame type
    by_frame = {}
    for r in all_query_results:
        frame = r.get("frame_type", "unknown")
        if frame not in by_frame:
            by_frame[frame] = {"count": 0, "correct": 0}
        by_frame[frame]["count"] += 1
        by_frame[frame]["correct"] += int(r["correct"])
    for frame in by_frame:
        by_frame[frame]["accuracy"] = by_frame[frame]["correct"] / by_frame[frame]["count"]

    # Per-image summary
    per_image = [
        {
            "image": r["image"],
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["num_queries"],
        }
        for r in all_results
    ]

    return {
        "model": {"provider": "qwen", "model_name": "Qwen/Qwen2.5-VL-7B-Instruct"},
        "summary": {
            "num_images": len(all_results),
            "total_queries": total_queries,
            "total_correct": total_correct,
            "overall_accuracy": total_correct / total_queries if total_queries else 0,
        },
        "by_task_type": by_task_type,
        "by_relation_axis": by_axis,
        "by_frame_type": by_frame,
        "per_image": per_image,
        "all_results": all_query_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run batch Qwen-VL evaluation")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory with spatial graphs")
    parser.add_argument("--output-dir", "-o", default="qwen_batch_eval", help="Output directory")
    parser.add_argument("--queries-per-bucket", type=int, default=4, help="Queries per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-images", type=int, help="Max images to process (for testing)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all spatial graphs
    graph_files = sorted(input_dir.glob("*_spatial_graph.json"))
    if args.max_images:
        graph_files = graph_files[:args.max_images]

    print(f"Found {len(graph_files)} spatial graphs")

    # Load model once
    model, processor = load_qwen_model()

    # Process each image
    all_results = []
    for idx, graph_path in enumerate(graph_files, 1):
        image_name = graph_path.stem.replace("_spatial_graph", "")
        print(f"\n[{idx}/{len(graph_files)}] Processing {image_name}...")

        try:
            result = evaluate_single_image(
                model, processor,
                str(graph_path),
                queries_per_bucket=args.queries_per_bucket,
                seed=args.seed,
            )
            all_results.append(result)

            print(f"  Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['num_queries']})")

            # Save individual result
            with open(output_dir / f"{image_name}_result.json", "w") as f:
                json.dump(result, f, indent=2)

            # Save progress
            progress = {
                "completed": idx,
                "total": len(graph_files),
                "current_overall": sum(r["correct"] for r in all_results) / sum(r["num_queries"] for r in all_results),
            }
            with open(output_dir / "progress.json", "w") as f:
                json.dump(progress, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATING RESULTS")
    print("=" * 60)

    aggregated = aggregate_results(all_results)

    # Save aggregated results
    with open(output_dir / "aggregated_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    print(f"\nOverall: {aggregated['summary']['overall_accuracy']:.1%} "
          f"({aggregated['summary']['total_correct']}/{aggregated['summary']['total_queries']})")

    print("\nBy Task Type:")
    for task, stats in sorted(aggregated["by_task_type"].items()):
        print(f"  {task}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['count']})")

    print("\nBy Relation Axis:")
    for axis, stats in sorted(aggregated["by_relation_axis"].items()):
        print(f"  {axis}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['count']})")

    print("\nBy Frame Type:")
    for frame, stats in sorted(aggregated["by_frame_type"].items()):
        print(f"  {frame}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['count']})")

    print(f"\nResults saved to: {output_dir}")
    print(f"Aggregated results: {output_dir / 'aggregated_results.json'}")


if __name__ == "__main__":
    main()
