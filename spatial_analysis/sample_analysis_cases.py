#!/usr/bin/env python3
"""
sample_analysis_cases.py - Sample cases for activation and attention analysis.

Samples stratified by:
- Task type (egocentric vs allocentric)
- Relation axis (left_right, front_behind, above_below, foreground_background)
- Correctness (success vs failure)

Usage:
    python sample_analysis_cases.py --input-dir qwen_batch_eval_50_full
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_all_results_with_images(results_dir: str) -> list:
    """Load all results with image information preserved."""
    results_path = Path(results_dir)
    all_results = []

    # Load per-image results to get image mapping
    for result_file in sorted(results_path.glob("*_result.json")):
        with open(result_file) as f:
            data = json.load(f)

        image_id = data["image"]

        for r in data["results"]:
            r["image_id"] = image_id
            all_results.append(r)

    return all_results


def sample_cases(all_results: list, samples_per_stratum: int = 3, seed: int = 42):
    """Sample cases stratified by task type, relation axis, and correctness."""
    random.seed(seed)

    # Group by (task_type, relation_axis, correct)
    strata = defaultdict(list)
    for r in all_results:
        key = (r["task_type"], r.get("relation_axis", "unknown"), r["correct"])
        strata[key].append(r)

    # Sample from each stratum
    sampled = []
    for key, cases in sorted(strata.items()):
        task, axis, correct = key
        n_sample = min(samples_per_stratum, len(cases))
        selected = random.sample(cases, n_sample)

        for case in selected:
            case["_stratum"] = {
                "task_type": task,
                "relation_axis": axis,
                "correct": correct,
            }
            sampled.append(case)

    return sampled


def summarize_sample(sampled: list):
    """Print summary of sampled cases."""
    print(f"Total sampled: {len(sampled)} cases\n")

    # By task type and correctness
    by_task_correct = defaultdict(lambda: {"success": 0, "failure": 0})
    for case in sampled:
        s = case["_stratum"]
        key = s["task_type"]
        if s["correct"]:
            by_task_correct[key]["success"] += 1
        else:
            by_task_correct[key]["failure"] += 1

    print("By Task Type:")
    for task in sorted(by_task_correct):
        stats = by_task_correct[task]
        print(f"  {task}: {stats['success']} success, {stats['failure']} failure")

    # By relation axis
    by_axis = defaultdict(int)
    for case in sampled:
        by_axis[case["_stratum"]["relation_axis"]] += 1

    print("\nBy Relation Axis:")
    for axis in sorted(by_axis):
        print(f"  {axis}: {by_axis[axis]}")

    # By image
    by_image = defaultdict(int)
    for case in sampled:
        by_image[case["image_id"]] += 1
    print(f"\nUnique images: {len(by_image)}")


def get_case_details(case: dict) -> dict:
    """Extract relevant details for analysis."""
    return {
        "query_id": case["query_id"],
        "image_id": case["image_id"],
        "task_type": case["task_type"],
        "frame_type": case["frame_type"],
        "relation_axis": case.get("relation_axis"),
        "prompt": case["prompt"],
        "ground_truth": case["ground_truth_answer"],
        "model_answer": case["model_answer"],
        "correct": case["correct"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", default="qwen_batch_eval_50_full",
                        help="Directory with per-image result files")
    parser.add_argument("--output", "-o", default="spatial_analysis/analysis_samples.json",
                        help="Output file")
    parser.add_argument("--samples-per-stratum", type=int, default=3,
                        help="Samples per category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load all results with image info
    all_results = load_all_results_with_images(args.input_dir)
    print(f"Loaded {len(all_results)} total results")

    # Sample
    sampled = sample_cases(
        all_results,
        samples_per_stratum=args.samples_per_stratum,
        seed=args.seed
    )

    summarize_sample(sampled)

    # Prepare output
    output = {
        "metadata": {
            "source": args.input_dir,
            "samples_per_stratum": args.samples_per_stratum,
            "seed": args.seed,
            "total_sampled": len(sampled),
        },
        "cases": [get_case_details(c) for c in sampled],
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSampled cases saved to: {output_path}")


if __name__ == "__main__":
    main()
