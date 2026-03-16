#!/usr/bin/env python3
"""
visualize_query_examples.py - Generate visual examples for each query category.

Creates a grid visualization showing:
- The labeled benchmark image
- Example prompts for each (task_type, frame_type, relation_axis) combination
- Model responses and correctness

Usage:
    python visualize_query_examples.py \
        --eval-results query_benchmark_outputs/living_room2__openai__gpt-4o/evaluation_results.json \
        --queries query_benchmark_outputs/living_room2__openai__gpt-4o/queries.json \
        --image photos/living_room2.jpg \
        --output query_examples_visualization.png
"""

import argparse
import json
import os
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def render_labeled_image(
    image_path: str,
    graph_path: Optional[str] = None,
    queries_data: Optional[Dict] = None,
) -> Image.Image:
    """Render image with bounding box labels from spatial graph or queries metadata."""
    image = Image.open(image_path).convert("RGB")

    # Try to load spatial graph for node info
    nodes = []
    if graph_path and Path(graph_path).exists():
        graph_data = load_json(graph_path)
        nodes = graph_data.get("nodes", [])

    if not nodes:
        # Return original image if no graph data
        return image

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

    # Color palette for different z-orders
    colors = [
        (255, 99, 71),   # tomato (foreground)
        (255, 165, 0),   # orange
        (255, 215, 0),   # gold
        (154, 205, 50),  # yellowgreen
        (60, 179, 113),  # mediumseagreen
        (70, 130, 180),  # steelblue
        (138, 43, 226),  # blueviolet (background)
    ]

    for node in nodes:
        bbox = node.get("bbox", [])
        if len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        name = node.get("name", "unknown")
        z_order = node.get("z_order", 0)

        # Pick color based on z-order
        color_idx = min(z_order, len(colors) - 1)
        color = colors[color_idx]

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label background
        label = f"{name}"
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        ly = max(0, y1 - th - 4)
        draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=(0, 0, 0, 200))
        draw.text((x1 + 3, ly + 2), label, fill=(255, 255, 255), font=font)

    return image


def wrap_text(text: str, width: int = 60) -> str:
    """Wrap text to specified width."""
    return "\n".join(textwrap.wrap(text, width=width))


def get_category_examples(
    eval_results: Dict,
    queries_data: Dict,
    examples_per_category: int = 2,
) -> Dict[Tuple[str, str, str], List[Dict]]:
    """Group evaluation results by (task_type, frame_type, relation_axis) and pick examples."""

    # Build query lookup
    query_lookup = {q["query_id"]: q for q in queries_data.get("queries", [])}

    # Group results
    groups = defaultdict(list)
    for result in eval_results.get("results", []):
        key = (
            result.get("task_type", "unknown"),
            result.get("frame_type", "unknown"),
            result.get("relation_axis", "unknown"),
        )
        # Merge query data with result
        query_id = result.get("query_id")
        query_info = query_lookup.get(query_id, {})
        merged = {**query_info, **result}
        groups[key].append(merged)

    # Pick examples: one correct, one incorrect if available
    examples = {}
    for key, items in groups.items():
        correct_items = [i for i in items if i.get("correct")]
        incorrect_items = [i for i in items if not i.get("correct")]

        selected = []
        if correct_items:
            selected.append(correct_items[0])
        if incorrect_items and len(selected) < examples_per_category:
            selected.append(incorrect_items[0])
        if len(selected) < examples_per_category and len(items) > len(selected):
            for item in items:
                if item not in selected:
                    selected.append(item)
                    if len(selected) >= examples_per_category:
                        break

        examples[key] = selected

    return examples


def prettify_label(label: str) -> str:
    """Convert snake_case to Title Case."""
    mapping = {
        "allocentric_qa": "Allocentric QA",
        "egocentric_qa": "Egocentric QA",
        "viewer_centered": "Viewer-Centered",
        "object_to_object": "Object-to-Object",
        "left_right": "Left / Right",
        "above_below": "Above / Below",
        "foreground_background": "Foreground / Background",
        "front_behind": "Front / Behind",
    }
    return mapping.get(label, label.replace("_", " ").title())


def create_example_panel(
    ax: plt.Axes,
    example: Dict,
    category_label: str,
) -> None:
    """Create a single example panel showing prompt and response."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Category header
    ax.text(
        5, 9.5, category_label,
        ha="center", va="top",
        fontsize=11, fontweight="bold",
        color="#2C3E50",
    )

    # Prompt (wrapped)
    prompt = example.get("prompt", "N/A")
    wrapped_prompt = wrap_text(prompt, width=50)
    ax.text(
        0.5, 8.5, f"Prompt:\n{wrapped_prompt}",
        ha="left", va="top",
        fontsize=8,
        color="#34495E",
        family="monospace",
    )

    # Ground truth and model answer
    gt = example.get("ground_truth_answer", "N/A")
    model_ans = example.get("model_answer", "N/A")
    correct = example.get("correct", False)

    # Color based on correctness
    result_color = "#27AE60" if correct else "#E74C3C"
    result_symbol = "✓" if correct else "✗"

    y_pos = 3.0
    ax.text(
        0.5, y_pos,
        f"Ground Truth: {gt}",
        ha="left", va="top",
        fontsize=9, fontweight="bold",
        color="#2C3E50",
    )
    ax.text(
        0.5, y_pos - 0.8,
        f"Model Answer: {model_ans}  {result_symbol}",
        ha="left", va="top",
        fontsize=9, fontweight="bold",
        color=result_color,
    )

    # Additional info
    info_text = []
    if example.get("anchor_object"):
        info_text.append(f"Anchor: {example['anchor_object']}")
    if example.get("reference_object"):
        info_text.append(f"Reference: {example['reference_object']}")
    if example.get("target_object"):
        info_text.append(f"Target: {example['target_object']}")
    if example.get("orientation"):
        info_text.append(f"Orientation: {example['orientation']}")

    if info_text:
        ax.text(
            0.5, y_pos - 2.0,
            " | ".join(info_text),
            ha="left", va="top",
            fontsize=7,
            color="#7F8C8D",
        )


def create_visualization(
    image_path: str,
    eval_results: Dict,
    queries_data: Dict,
    graph_path: Optional[str],
    output_path: str,
    examples_per_category: int = 1,
) -> None:
    """Create the full visualization with image and example panels."""

    # Get examples per category
    examples = get_category_examples(eval_results, queries_data, examples_per_category)

    # Sort categories
    sorted_categories = sorted(examples.keys())
    n_categories = len(sorted_categories)

    # Layout: image on left, examples in grid on right
    n_cols = 2
    n_rows = (n_categories + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(18, max(10, n_rows * 3.5)))

    # Create grid spec: image takes left 35%, examples take right 65%
    gs = fig.add_gridspec(
        n_rows, 3,
        width_ratios=[1.2, 1, 1],
        height_ratios=[1] * n_rows,
        hspace=0.3,
        wspace=0.2,
    )

    # Load and render labeled image
    labeled_image = render_labeled_image(image_path, graph_path, queries_data)

    # Image panel (spans all rows on left)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(labeled_image)
    ax_img.set_title("Benchmark Image\n(with labeled objects)", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    # Model info
    model_info = eval_results.get("model", {})
    overall = eval_results.get("overall", {})
    model_text = f"Model: {model_info.get('model_name', 'N/A')} ({model_info.get('provider', 'N/A')})"
    accuracy_text = f"Overall Accuracy: {overall.get('accuracy', 0):.1%} ({overall.get('correct', 0)}/{overall.get('count', 0)})"

    fig.text(
        0.18, 0.02,
        f"{model_text}\n{accuracy_text}",
        ha="center", va="bottom",
        fontsize=10,
        color="#2C3E50",
    )

    # Example panels
    for idx, category in enumerate(sorted_categories):
        row = idx // n_cols
        col = idx % n_cols + 1  # +1 because image is at col 0

        ax = fig.add_subplot(gs[row, col])

        category_label = f"{prettify_label(category[0])}\n{prettify_label(category[1])} | {prettify_label(category[2])}"

        # Get first example
        if examples[category]:
            create_example_panel(ax, examples[category][0], category_label)
        else:
            ax.text(5, 5, "No examples", ha="center", va="center")
            ax.axis("off")

    # Title
    fig.suptitle(
        "Query Benchmark Examples by Category",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Save
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved visualization to: {output_path}")


def create_detailed_per_axis_visualization(
    image_path: str,
    eval_results: Dict,
    queries_data: Dict,
    graph_path: Optional[str],
    output_dir: str,
) -> None:
    """Create separate detailed visualizations for each relation axis."""

    examples = get_category_examples(eval_results, queries_data, examples_per_category=3)

    # Group by relation axis
    by_axis = defaultdict(list)
    for category, items in examples.items():
        axis = category[2]
        by_axis[axis].extend([(category, item) for item in items])

    # Load labeled image once
    labeled_image = render_labeled_image(image_path, graph_path, queries_data)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for axis, category_items in by_axis.items():
        n_examples = len(category_items)
        if n_examples == 0:
            continue

        # Create figure for this axis
        fig = plt.figure(figsize=(16, 4 + n_examples * 2.5))

        gs = fig.add_gridspec(
            n_examples + 1, 2,
            width_ratios=[1, 1.5],
            height_ratios=[1.5] + [1] * n_examples,
            hspace=0.4,
            wspace=0.3,
        )

        # Image panel
        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(labeled_image)
        ax_img.set_title("Benchmark Image", fontsize=11, fontweight="bold")
        ax_img.axis("off")

        # Axis info panel
        ax_info = fig.add_subplot(gs[0, 1])
        ax_info.axis("off")

        model_info = eval_results.get("model", {})
        overall = eval_results.get("overall", {})

        info_text = f"""
Relation Axis: {prettify_label(axis)}

Model: {model_info.get('model_name', 'N/A')} ({model_info.get('provider', 'N/A')})
Overall Accuracy: {overall.get('accuracy', 0):.1%}

Examples shown below demonstrate queries testing
the "{prettify_label(axis)}" spatial relation across
different task types and frame types.
        """.strip()

        ax_info.text(
            0.1, 0.9, info_text,
            ha="left", va="top",
            fontsize=10,
            transform=ax_info.transAxes,
            family="sans-serif",
        )

        # Example rows
        for idx, (category, example) in enumerate(category_items):
            ax = fig.add_subplot(gs[idx + 1, :])
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 2)
            ax.axis("off")

            # Category
            cat_label = f"{prettify_label(category[0])} | {prettify_label(category[1])}"
            ax.text(0.1, 1.8, cat_label, fontsize=10, fontweight="bold", color="#2C3E50")

            # Prompt
            prompt = example.get("prompt", "N/A")
            wrapped = wrap_text(prompt, width=90)
            ax.text(0.1, 1.4, wrapped, fontsize=8, color="#34495E", family="monospace", va="top")

            # Results
            gt = example.get("ground_truth_answer", "N/A")
            model_ans = example.get("model_answer", "N/A")
            correct = example.get("correct", False)

            result_color = "#27AE60" if correct else "#E74C3C"
            symbol = "✓ CORRECT" if correct else "✗ INCORRECT"

            ax.text(
                0.1, 0.3,
                f"Ground Truth: {gt}    |    Model: {model_ans}    |    {symbol}",
                fontsize=9,
                color=result_color,
                fontweight="bold",
            )

            # Separator line
            ax.axhline(y=0.1, xmin=0.01, xmax=0.99, color="#BDC3C7", linewidth=0.5)

        # Title
        fig.suptitle(
            f"Query Examples: {prettify_label(axis)} Axis",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        # Save
        output_path = output_dir / f"examples_{axis}.png"
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize query benchmark examples")
    parser.add_argument("--eval-results", "-e", required=True, help="Path to evaluation_results.json")
    parser.add_argument("--queries", "-q", required=True, help="Path to queries.json")
    parser.add_argument("--image", "-i", required=True, help="Path to source image")
    parser.add_argument("--graph", "-g", help="Path to spatial_graph.json (optional)")
    parser.add_argument("--output", "-o", default="query_examples.png", help="Output image path")
    parser.add_argument("--per-axis", action="store_true", help="Generate separate visualization per axis")
    parser.add_argument("--output-dir", default="query_visualizations", help="Output directory for per-axis visualizations")
    args = parser.parse_args()

    eval_results = load_json(args.eval_results)
    queries_data = load_json(args.queries)

    # Try to find graph path from queries metadata
    graph_path = args.graph
    if not graph_path:
        graph_path = queries_data.get("metadata", {}).get("graph_path")

    if args.per_axis:
        create_detailed_per_axis_visualization(
            args.image,
            eval_results,
            queries_data,
            graph_path,
            args.output_dir,
        )
    else:
        create_visualization(
            args.image,
            eval_results,
            queries_data,
            graph_path,
            args.output,
        )


if __name__ == "__main__":
    main()
