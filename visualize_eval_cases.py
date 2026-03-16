#!/usr/bin/env python3
"""
visualize_eval_cases.py - Visualize success and failure cases from evaluation results.
"""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


def load_json(path):
    with open(path) as f:
        return json.load(f)


def create_case_visualization(
    image_path: str,
    results_openai: dict,
    results_claude: dict,
    output_dir: str,
):
    """Create visualizations comparing success/failure cases between models."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    openai_data = results_openai["results"]
    claude_data = results_claude["results"]

    # Build lookup by query content (since query_ids differ)
    def get_key(r):
        return (r["task_type"], r["frame_type"], r["relation_axis"], r["ground_truth_answer"])

    # Find matching queries
    openai_by_key = {get_key(r): r for r in openai_data}
    claude_by_key = {get_key(r): r for r in claude_data}

    # Categorize cases
    both_correct = []
    both_wrong = []
    openai_only = []  # OpenAI correct, Claude wrong
    claude_only = []  # Claude correct, OpenAI wrong

    for key, o_result in openai_by_key.items():
        if key not in claude_by_key:
            continue
        c_result = claude_by_key[key]

        if o_result["correct"] and c_result["correct"]:
            both_correct.append((o_result, c_result))
        elif not o_result["correct"] and not c_result["correct"]:
            both_wrong.append((o_result, c_result))
        elif o_result["correct"] and not c_result["correct"]:
            openai_only.append((o_result, c_result))
        else:
            claude_only.append((o_result, c_result))

    # Load image
    img = Image.open(image_path)

    # Create visualization for each category
    def create_cases_figure(cases, title, filename, max_cases=4):
        if not cases:
            return

        n_cases = min(len(cases), max_cases)
        fig = plt.figure(figsize=(16, 4 + 3 * n_cases))
        gs = GridSpec(n_cases + 1, 2, figure=fig, height_ratios=[1.5] + [1] * n_cases,
                      hspace=0.4, wspace=0.1)

        # Image at top
        ax_img = fig.add_subplot(gs[0, :])
        ax_img.imshow(img)
        ax_img.set_title("Benchmark Image", fontsize=11, fontweight="bold")
        ax_img.axis("off")

        for i, (o_result, c_result) in enumerate(cases[:n_cases]):
            ax = fig.add_subplot(gs[i + 1, :])
            ax.axis("off")

            # Format the case
            prompt = o_result["prompt"]
            if len(prompt) > 120:
                prompt = prompt[:120] + "..."

            text = f"Query {i+1}: {o_result['task_type']} | {o_result['frame_type']} | {o_result['relation_axis']}\n"
            text += f"─" * 80 + "\n"
            text += f"Prompt: {prompt}\n\n"
            text += f"Ground Truth: {o_result['ground_truth_answer']}\n\n"
            text += f"GPT-5-mini:      {o_result['model_answer']:<20} "
            text += f"{'✓ CORRECT' if o_result['correct'] else '✗ WRONG'}\n"
            text += f"Claude Haiku:    {c_result['model_answer']:<20} "
            text += f"{'✓ CORRECT' if c_result['correct'] else '✗ WRONG'}"

            # Color based on correctness
            bbox_color = "#d4edda" if o_result["correct"] and c_result["correct"] else \
                         "#f8d7da" if not o_result["correct"] and not c_result["correct"] else \
                         "#fff3cd"

            ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=9,
                    verticalalignment="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=bbox_color, alpha=0.8))

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {output_dir / filename}")

    # Generate visualizations
    create_cases_figure(both_correct,
                        f"Both Models Correct ({len(both_correct)} cases)",
                        "cases_both_correct.png")

    create_cases_figure(both_wrong,
                        f"Both Models Wrong ({len(both_wrong)} cases)",
                        "cases_both_wrong.png")

    create_cases_figure(openai_only,
                        f"GPT-5-mini Correct, Claude Wrong ({len(openai_only)} cases)",
                        "cases_openai_wins.png")

    create_cases_figure(claude_only,
                        f"Claude Correct, GPT-5-mini Wrong ({len(claude_only)} cases)",
                        "cases_claude_wins.png")

    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    categories = ["Both Correct", "Both Wrong", "GPT-5 Only", "Claude Only"]
    counts = [len(both_correct), len(both_wrong), len(openai_only), len(claude_only)]
    colors = ["#28a745", "#dc3545", "#007bff", "#6f42c1"]

    # Pie chart
    ax = axes[0, 0]
    ax.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title("Case Distribution", fontweight="bold")

    # Bar chart
    ax = axes[0, 1]
    ax.bar(categories, counts, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Case Counts", fontweight="bold")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")

    # Accuracy by axis comparison
    ax = axes[1, 0]
    axes_names = ["left_right", "above_below", "fg_bg", "front_behind"]
    openai_accs = []
    claude_accs = []

    for axis in ["left_right", "above_below", "foreground_background", "front_behind"]:
        o_axis = [r for r in openai_data if r["relation_axis"] == axis]
        c_axis = [r for r in claude_data if r["relation_axis"] == axis]
        openai_accs.append(sum(1 for r in o_axis if r["correct"]) / len(o_axis) * 100 if o_axis else 0)
        claude_accs.append(sum(1 for r in c_axis if r["correct"]) / len(c_axis) * 100 if c_axis else 0)

    x = range(len(axes_names))
    width = 0.35
    ax.bar([i - width/2 for i in x], openai_accs, width, label="GPT-5-mini", color="#007bff")
    ax.bar([i + width/2 for i in x], claude_accs, width, label="Claude Haiku", color="#6f42c1")
    ax.set_ylabel("Accuracy %")
    ax.set_xticks(x)
    ax.set_xticklabels(axes_names, rotation=15)
    ax.legend()
    ax.set_title("Accuracy by Relation Axis", fontweight="bold")
    ax.set_ylim(0, 105)

    # Accuracy by task type
    ax = axes[1, 1]
    task_names = ["egocentric", "allocentric"]
    openai_task = []
    claude_task = []

    for task in ["egocentric_qa", "allocentric_qa"]:
        o_task = [r for r in openai_data if r["task_type"] == task]
        c_task = [r for r in claude_data if r["task_type"] == task]
        openai_task.append(sum(1 for r in o_task if r["correct"]) / len(o_task) * 100 if o_task else 0)
        claude_task.append(sum(1 for r in c_task if r["correct"]) / len(c_task) * 100 if c_task else 0)

    x = range(len(task_names))
    ax.bar([i - width/2 for i in x], openai_task, width, label="GPT-5-mini", color="#007bff")
    ax.bar([i + width/2 for i in x], claude_task, width, label="Claude Haiku", color="#6f42c1")
    ax.set_ylabel("Accuracy %")
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.legend()
    ax.set_title("Accuracy by Task Type", fontweight="bold")
    ax.set_ylim(0, 105)

    plt.suptitle("Evaluation Summary: GPT-5-mini vs Claude Haiku 4.5", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "summary_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_dir / 'summary_comparison.png'}")

    print(f"\nCase counts:")
    print(f"  Both correct: {len(both_correct)}")
    print(f"  Both wrong: {len(both_wrong)}")
    print(f"  GPT-5-mini only: {len(openai_only)}")
    print(f"  Claude only: {len(claude_only)}")


if __name__ == "__main__":
    create_case_visualization(
        image_path="spatial_outputs_coco_50/000000000139_spatial_viz.png",
        results_openai=load_json("eval_outputs/000000000139__openai__openai/evaluation_results.json"),
        results_claude=load_json("eval_outputs/000000000139__claude__claude/evaluation_results.json"),
        output_dir="eval_visualizations",
    )
