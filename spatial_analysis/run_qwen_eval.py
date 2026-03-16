#!/usr/bin/env python3
"""
run_qwen_eval.py - Run Qwen-VL baseline evaluation for spatial reasoning.

This script evaluates Qwen2.5-VL-7B on the spatial benchmark and collects
results for comparison with closed-source models (GPT-5-mini, Claude Haiku).

Usage:
    python run_qwen_eval.py --graph <spatial_graph.json> --output-dir <output>
"""

import argparse
import json
import time
from pathlib import Path
import sys

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
    """Generate a response from Qwen for a given image and prompt."""
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


def evaluate_queries(model, processor, image_path: str, queries: list, output_path: Path = None) -> dict:
    """Evaluate Qwen on a list of queries."""
    results = []
    correct = 0
    total = len(queries)

    for idx, query in enumerate(queries, start=1):
        print(
            f"[{idx}/{total}] {query['task_type']} | {query['frame_type']} | {query['relation_axis']}",
            end=" ... ",
            flush=True
        )

        try:
            response = generate_response(
                model, processor,
                image_path,
                query["prompt"],
                QUERY_SYSTEM_PROMPT
            )
        except Exception as e:
            print(f"ERROR: {e}")
            response = "ERROR"

        model_answer = normalize_answer(response, query.get("candidate_answers"))
        gt = normalize_answer(query["ground_truth_answer"], query.get("candidate_answers"))
        is_correct = model_answer == gt

        if is_correct:
            correct += 1
            print(f"CORRECT ({response})")
        else:
            print(f"WRONG (got: {response}, expected: {gt})")

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

        # Save partial results
        if output_path:
            partial = {
                "model": {"provider": "qwen", "model_name": "Qwen/Qwen2.5-VL-7B-Instruct"},
                "overall": {
                    "count": len(results),
                    "correct": correct,
                    "accuracy": correct / len(results) if results else 0.0,
                    "partial": True,
                },
                "results": results,
            }
            output_path.write_text(json.dumps(partial, indent=2))

    # Compute breakdown by type
    by_type = {}
    for row in results:
        bucket = by_type.setdefault(row["task_type"], {"count": 0, "correct": 0})
        bucket["count"] += 1
        bucket["correct"] += int(row["correct"])
    for bucket in by_type.values():
        bucket["accuracy"] = bucket["correct"] / bucket["count"] if bucket["count"] else 0.0

    return {
        "model": {"provider": "qwen", "model_name": "Qwen/Qwen2.5-VL-7B-Instruct"},
        "overall": {
            "count": len(results),
            "correct": correct,
            "accuracy": correct / len(results) if results else 0.0,
            "partial": False,
        },
        "by_task_type": by_type,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Qwen-VL evaluation")
    parser.add_argument("--graph", "-g", required=True, help="Path to spatial_graph.json")
    parser.add_argument("--output-dir", "-o", default="qwen_eval_outputs", help="Output directory")
    parser.add_argument("--queries-per-bucket", type=int, default=4, help="Queries per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate queries
    print("Generating queries...")
    generated = generate_queries(
        args.graph,
        include_object_retrieval=False,
        queries_per_bucket=args.queries_per_bucket,
        seed=args.seed,
    )
    queries = generated["queries"]
    print(f"Generated {len(queries)} queries")

    # Save queries
    queries_path = output_dir / "queries.json"
    with open(queries_path, "w") as f:
        json.dump(generated, f, indent=2)
    print(f"Queries saved to: {queries_path}")

    # Resolve benchmark image
    graph_data = load_graph(args.graph)
    existing_labeled = resolve_stage1_labeled_image(args.graph)
    if existing_labeled:
        benchmark_image = str(existing_labeled)
    else:
        benchmark_image = str(resolve_image_path(graph_data, args.graph))
    print(f"Using benchmark image: {benchmark_image}")

    # Load model
    model, processor = load_qwen_model()

    # Run evaluation
    print("\n" + "=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)

    eval_path = output_dir / "evaluation_results.json"
    evaluation = evaluate_queries(model, processor, benchmark_image, queries, eval_path)

    # Save final results
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Overall accuracy: {evaluation['overall']['accuracy']:.1%} "
          f"({evaluation['overall']['correct']}/{evaluation['overall']['count']})")
    print("\nBreakdown by task type:")
    for task_type, stats in evaluation["by_task_type"].items():
        print(f"  {task_type}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['count']})")
    print(f"\nResults saved to: {eval_path}")


if __name__ == "__main__":
    main()
