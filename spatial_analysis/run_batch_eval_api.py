#!/usr/bin/env python3
"""
run_batch_eval_api.py - Run batch evaluation using OpenAI or Anthropic APIs.

This script evaluates VLMs on all spatial graphs in a directory using API calls.

Usage:
    python run_batch_eval_api.py --provider openai --model gpt-5-mini-2025-08-07 --input-dir spatial_outputs_coco_50 --output-dir gpt5_mini_batch_eval_50
    python run_batch_eval_api.py --provider anthropic --model claude-haiku-4-5 --input-dir spatial_outputs_coco_50 --output-dir claude_haiku_batch_eval_50
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

# Load environment variables from .env file (override existing env vars)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

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
If the valid answer choices are shown in the prompt, output exactly one of them and nothing else."""


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


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get media type from image path."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")


def generate_response_openai(client, model: str, image_path: str, prompt: str, system_prompt: str = "") -> str:
    """Generate response using OpenAI API."""
    base64_image = encode_image_base64(image_path)
    media_type = get_image_media_type(image_path)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{base64_image}"}
            },
            {"type": "text", "text": prompt}
        ]
    })

    # GPT-5-mini uses internal reasoning tokens, so we need more tokens
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=1024,
    )

    content = response.choices[0].message.content
    if content is None:
        return ""
    return content.strip()


def generate_response_anthropic(client, model: str, image_path: str, prompt: str, system_prompt: str = "") -> str:
    """Generate response using Anthropic API."""
    base64_image = encode_image_base64(image_path)
    media_type = get_image_media_type(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image,
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]

    response = client.messages.create(
        model=model,
        max_tokens=128,
        system=system_prompt if system_prompt else "",
        messages=messages,
    )

    return response.content[0].text.strip()


def evaluate_single_image(
    client,
    provider: str,
    model: str,
    graph_path: str,
    queries_per_bucket: int = 4,
    seed: int = 42,
    rate_limit_delay: float = 0.1,
):
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

    generate_fn = generate_response_openai if provider == "openai" else generate_response_anthropic

    for i, query in enumerate(queries):
        try:
            response = generate_fn(
                client, model,
                benchmark_image,
                query["prompt"],
                QUERY_SYSTEM_PROMPT
            )
            time.sleep(rate_limit_delay)  # Rate limiting
        except Exception as e:
            print(f"    Query {i+1} error: {e}")
            response = "ERROR"
            time.sleep(1)  # Longer delay on error

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


def aggregate_results(all_results: list, provider: str, model: str) -> dict:
    """Aggregate results across all images."""
    total_queries = 0
    total_correct = 0
    all_query_results = []

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

    # Hierarchical breakdown: task_type x frame_type x relation_axis
    hierarchical = {}
    for r in all_query_results:
        key = (r["task_type"], r["frame_type"], r.get("relation_axis", "unknown"))
        if key not in hierarchical:
            hierarchical[key] = {"count": 0, "correct": 0}
        hierarchical[key]["count"] += 1
        hierarchical[key]["correct"] += int(r["correct"])

    hierarchical_list = []
    for (task, frame, axis), stats in sorted(hierarchical.items()):
        hierarchical_list.append({
            "task_type": task,
            "frame_type": frame,
            "relation_axis": axis,
            "count": stats["count"],
            "correct": stats["correct"],
            "accuracy": stats["correct"] / stats["count"] if stats["count"] > 0 else 0,
        })

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
        "model": {"provider": provider, "model_name": model},
        "summary": {
            "num_images": len(all_results),
            "total_queries": total_queries,
            "total_correct": total_correct,
            "overall_accuracy": total_correct / total_queries if total_queries else 0,
        },
        "by_task_type": by_task_type,
        "by_relation_axis": by_axis,
        "by_frame_type": by_frame,
        "hierarchical": hierarchical_list,
        "per_image": per_image,
        "all_results": all_query_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run batch VLM evaluation via API")
    parser.add_argument("--provider", "-p", required=True, choices=["openai", "anthropic"],
                        help="API provider")
    parser.add_argument("--model", "-m", required=True,
                        help="Model name (e.g., gpt-4o-mini, claude-3-5-haiku-20241022)")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory with spatial graphs")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--queries-per-bucket", type=int, default=4, help="Queries per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-images", type=int, help="Max images to process")
    parser.add_argument("--rate-limit-delay", type=float, default=0.1,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # Initialize client
    if args.provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        print(f"Using OpenAI API with model: {args.model}")
    else:
        import anthropic
        client = anthropic.Anthropic()
        print(f"Using Anthropic API with model: {args.model}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all spatial graphs
    graph_files = sorted(input_dir.glob("*_spatial_graph.json"))
    if args.max_images:
        graph_files = graph_files[:args.max_images]

    print(f"Found {len(graph_files)} spatial graphs")

    # Check for existing results (resume support)
    all_results = []
    processed_images = set()

    if args.resume:
        for result_file in output_dir.glob("*_result.json"):
            try:
                with open(result_file) as f:
                    result = json.load(f)
                all_results.append(result)
                processed_images.add(result["image"])
                print(f"  Loaded existing: {result['image']}")
            except Exception as e:
                print(f"  Error loading {result_file}: {e}")
        print(f"Resuming with {len(all_results)} existing results")

    # Process each image
    for idx, graph_path in enumerate(graph_files, 1):
        image_name = graph_path.stem.replace("_spatial_graph", "")

        if image_name in processed_images:
            continue

        print(f"\n[{idx}/{len(graph_files)}] Processing {image_name}...")

        try:
            result = evaluate_single_image(
                client,
                args.provider,
                args.model,
                str(graph_path),
                queries_per_bucket=args.queries_per_bucket,
                seed=args.seed,
                rate_limit_delay=args.rate_limit_delay,
            )
            all_results.append(result)

            print(f"  Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['num_queries']})")

            # Save individual result
            with open(output_dir / f"{image_name}_result.json", "w") as f:
                json.dump(result, f, indent=2)

            # Save progress
            current_correct = sum(r["correct"] for r in all_results)
            current_total = sum(r["num_queries"] for r in all_results)
            progress = {
                "completed": len(all_results),
                "total": len(graph_files),
                "current_overall": current_correct / current_total if current_total > 0 else 0,
            }
            with open(output_dir / "progress.json", "w") as f:
                json.dump(progress, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATING RESULTS")
    print("=" * 60)

    aggregated = aggregate_results(all_results, args.provider, args.model)

    # Save aggregated results
    with open(output_dir / "aggregated_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    print(f"\nModel: {args.provider}/{args.model}")
    print(f"Overall: {aggregated['summary']['overall_accuracy']:.1%} "
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
