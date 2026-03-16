#!/usr/bin/env python3
"""
visualize_query_detailed.py - Generate detailed visualizations for ALL query dimension combinations.

Creates comprehensive visualizations showing examples for each combination of:
- Task Type: egocentric_qa, allocentric_qa
- Frame Type: viewer_centered, object_to_object
- Query Subtype: binary_relation, relation_mcq, object_retrieval
- Relation Axis: left_right, above_below, foreground_background, front_behind
- Orientation (allocentric only): camera, away, left, right

Usage:
    python visualize_query_detailed.py \
        --queries query_benchmark_outputs/coco_000000000139/000000000139__queries/queries.json \
        --graph spatial_outputs_coco/000000000139_spatial_graph.json \
        --output-dir query_visualizations_detailed/
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
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def render_labeled_image(image_path: str, graph_path: Optional[str] = None) -> Image.Image:
    """Render image with bounding box labels from spatial graph."""
    image = Image.open(image_path).convert("RGB")

    nodes = []
    if graph_path and Path(graph_path).exists():
        graph_data = load_json(graph_path)
        nodes = graph_data.get("nodes", [])

    if not nodes:
        return image

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

    colors = [
        (255, 99, 71), (255, 165, 0), (255, 215, 0), (154, 205, 50),
        (60, 179, 113), (70, 130, 180), (138, 43, 226),
    ]

    for node in nodes:
        bbox = node.get("bbox", [])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        name = node.get("name", "unknown")
        z_order = node.get("z_order", 0)
        color = colors[min(z_order, len(colors) - 1)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{name}"
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        ly = max(0, y1 - th - 4)
        draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=(0, 0, 0, 200))
        draw.text((x1 + 3, ly + 2), label, fill=(255, 255, 255), font=font)

    return image


def wrap_text(text: str, width: int = 80) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def prettify(label: str) -> str:
    mapping = {
        "allocentric_qa": "Allocentric QA",
        "egocentric_qa": "Egocentric QA",
        "viewer_centered": "Viewer-Centered",
        "object_to_object": "Object-to-Object",
        "binary_relation": "Binary Relation",
        "relation_mcq": "Relation MCQ",
        "object_retrieval": "Object Retrieval",
        "left_right": "Left / Right",
        "above_below": "Above / Below",
        "foreground_background": "Foreground / Background",
        "front_behind": "Front / Behind",
        "camera": "Facing Camera",
        "away": "Facing Away",
        "left": "Facing Left",
        "right": "Facing Right",
    }
    return mapping.get(label, label.replace("_", " ").title())


def group_queries_by_all_dimensions(queries: List[Dict]) -> Dict[Tuple, List[Dict]]:
    """Group queries by (task_type, frame_type, query_subtype, relation_axis, orientation)."""
    groups = defaultdict(list)
    for q in queries:
        key = (
            q.get("task_type", "unknown"),
            q.get("frame_type", "unknown"),
            q.get("query_subtype", "unknown"),
            q.get("relation_axis", "unknown"),
            q.get("orientation") or "none",
        )
        groups[key].append(q)
    return dict(groups)


def create_dimension_summary_table(queries: List[Dict], output_path: Path) -> None:
    """Create a summary table showing query counts per dimension combination."""
    groups = group_queries_by_all_dimensions(queries)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis("off")

    # Build table data
    headers = ["Task Type", "Frame Type", "Query Subtype", "Relation Axis", "Orientation", "Count", "Example"]
    rows = []

    for key in sorted(groups.keys()):
        task_type, frame_type, query_subtype, relation_axis, orientation = key
        count = len(groups[key])
        example_prompt = groups[key][0].get("prompt", "")[:60] + "..."
        rows.append([
            prettify(task_type),
            prettify(frame_type),
            prettify(query_subtype),
            prettify(relation_axis),
            prettify(orientation) if orientation != "none" else "-",
            str(count),
            example_prompt,
        ])

    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        loc="center",
        colWidths=[0.12, 0.12, 0.12, 0.14, 0.1, 0.06, 0.34],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    # Style header
    for j, header in enumerate(headers):
        cell = table[(0, j)]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#ECF0F1")
            else:
                cell.set_facecolor("#FFFFFF")

    plt.title(f"Query Dimension Summary ({len(queries)} total queries)", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def create_task_type_comparison(
    queries: List[Dict],
    labeled_image: Image.Image,
    output_path: Path,
) -> None:
    """Create visualization comparing Egocentric vs Allocentric queries."""

    ego_queries = [q for q in queries if q.get("task_type") == "egocentric_qa"]
    allo_queries = [q for q in queries if q.get("task_type") == "allocentric_qa"]

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.2)

    # Image
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(labeled_image)
    ax_img.set_title("Benchmark Image with Labeled Objects", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    # Egocentric examples
    ax_ego = fig.add_subplot(gs[1, 0])
    ax_ego.axis("off")
    ax_ego.set_title("EGOCENTRIC QA\n(Viewer's perspective)", fontsize=11, fontweight="bold", color="#2980B9")

    ego_text = "Egocentric queries ask about spatial relations from the VIEWER's perspective.\n"
    ego_text += "The viewer is assumed to be looking at the scene from the camera position.\n\n"

    for i, q in enumerate(ego_queries[:4]):
        ego_text += f"Example {i+1}:\n"
        ego_text += f"  Prompt: {wrap_text(q['prompt'], 70)}\n"
        ego_text += f"  Ground Truth: {q['ground_truth_answer']}\n"
        ego_text += f"  Frame: {prettify(q['frame_type'])} | Axis: {prettify(q['relation_axis'])}\n\n"

    ax_ego.text(0.02, 0.98, ego_text, transform=ax_ego.transAxes, fontsize=8,
                verticalalignment="top", family="monospace")

    # Allocentric examples
    ax_allo = fig.add_subplot(gs[1, 1])
    ax_allo.axis("off")
    ax_allo.set_title("ALLOCENTRIC QA\n(Object-centered perspective)", fontsize=11, fontweight="bold", color="#E74C3C")

    allo_text = "Allocentric queries require MENTAL ROTATION to adopt an object's viewpoint.\n"
    allo_text += "The viewer must imagine standing at an anchor object facing a direction.\n\n"

    for i, q in enumerate(allo_queries[:4]):
        allo_text += f"Example {i+1}:\n"
        allo_text += f"  Prompt: {wrap_text(q['prompt'], 70)}\n"
        allo_text += f"  Ground Truth: {q['ground_truth_answer']}\n"
        allo_text += f"  Orientation: {prettify(q.get('orientation', 'none'))}\n\n"

    ax_allo.text(0.02, 0.98, allo_text, transform=ax_allo.transAxes, fontsize=8,
                 verticalalignment="top", family="monospace")

    # Key differences
    ax_diff = fig.add_subplot(gs[2, :])
    ax_diff.axis("off")
    ax_diff.set_title("Key Differences", fontsize=11, fontweight="bold")

    diff_text = """
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                           EGOCENTRIC                    vs            ALLOCENTRIC               │
    ├─────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  Reference Frame:         Camera/Viewer position        │  Anchor object's position            │
    │  Mental Rotation:         NOT required                  │  REQUIRED                            │
    │  Orientation:             Fixed (viewer facing scene)   │  Variable (camera/away/left/right)   │
    │  Complexity:              Lower                         │  Higher                              │
    │  Example Phrasing:        "Is X left of you?"           │  "Stand at Y, face camera. Is X..."  │
    └─────────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    ax_diff.text(0.5, 0.5, diff_text, transform=ax_diff.transAxes, fontsize=9,
                 ha="center", va="center", family="monospace")

    plt.suptitle("Task Type Comparison: Egocentric vs Allocentric", fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def create_frame_type_comparison(
    queries: List[Dict],
    labeled_image: Image.Image,
    output_path: Path,
) -> None:
    """Create visualization comparing Viewer-Centered vs Object-to-Object frames."""

    vc_queries = [q for q in queries if q.get("frame_type") == "viewer_centered"]
    o2o_queries = [q for q in queries if q.get("frame_type") == "object_to_object"]

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], hspace=0.3, wspace=0.2)

    # Image
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(labeled_image)
    ax_img.set_title("Benchmark Image", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    # Viewer-Centered
    ax_vc = fig.add_subplot(gs[1, 0])
    ax_vc.axis("off")
    ax_vc.set_title("VIEWER-CENTERED Frame\n(Relation to viewer/anchor)", fontsize=11, fontweight="bold", color="#27AE60")

    vc_text = "Queries about a SINGLE target object's relation to the viewer or anchor.\n\n"
    for i, q in enumerate(vc_queries[:5]):
        vc_text += f"{i+1}. {q['prompt'][:80]}...\n"
        vc_text += f"   → Target: {q.get('target_object')} | GT: {q['ground_truth_answer']}\n\n"

    ax_vc.text(0.02, 0.98, vc_text, transform=ax_vc.transAxes, fontsize=8,
               verticalalignment="top", family="monospace")

    # Object-to-Object
    ax_o2o = fig.add_subplot(gs[1, 1])
    ax_o2o.axis("off")
    ax_o2o.set_title("OBJECT-TO-OBJECT Frame\n(Relation between two objects)", fontsize=11, fontweight="bold", color="#9B59B6")

    o2o_text = "Queries about the relation between TWO objects (reference and target).\n\n"
    for i, q in enumerate(o2o_queries[:5]):
        o2o_text += f"{i+1}. {q['prompt'][:80]}...\n"
        o2o_text += f"   → Ref: {q.get('reference_object')} | Target: {q.get('target_object')} | GT: {q['ground_truth_answer']}\n\n"

    ax_o2o.text(0.02, 0.98, o2o_text, transform=ax_o2o.transAxes, fontsize=8,
                verticalalignment="top", family="monospace")

    plt.suptitle("Frame Type Comparison: Viewer-Centered vs Object-to-Object", fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def create_query_subtype_comparison(
    queries: List[Dict],
    labeled_image: Image.Image,
    output_path: Path,
) -> None:
    """Create visualization comparing Binary Relation and Relation MCQ."""

    binary = [q for q in queries if q.get("query_subtype") == "binary_relation"]
    mcq = [q for q in queries if q.get("query_subtype") == "relation_mcq"]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.8, 1.2], hspace=0.3, wspace=0.2)

    # Image
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(labeled_image)
    ax_img.set_title("Benchmark Image", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    # Binary Relation
    ax_bin = fig.add_subplot(gs[1, 0])
    ax_bin.axis("off")
    ax_bin.set_title("BINARY RELATION\n(Two-choice question)", fontsize=11, fontweight="bold", color="#E74C3C")

    bin_text = "Direct question with exactly 2 answer options.\n"
    bin_text += "Model must choose: left/right, above/below, etc.\n\n"
    bin_text += "Examples:\n"
    for i, q in enumerate(binary[:5]):
        bin_text += f"\n{i+1}. {wrap_text(q['prompt'], 50)}\n"
        bin_text += f"   Choices: {q['candidate_answers']}\n"
        bin_text += f"   Ground Truth: {q['ground_truth_answer']}\n"

    ax_bin.text(0.02, 0.98, bin_text, transform=ax_bin.transAxes, fontsize=8,
                verticalalignment="top", family="monospace")

    # Relation MCQ
    ax_mcq = fig.add_subplot(gs[1, 1])
    ax_mcq.axis("off")
    ax_mcq.set_title("RELATION MCQ\n(Multiple choice format)", fontsize=11, fontweight="bold", color="#3498DB")

    mcq_text = "Same relation question framed as explicit MCQ.\n"
    mcq_text += "Reinforces structured answer format.\n\n"
    mcq_text += "Examples:\n"
    for i, q in enumerate(mcq[:5]):
        mcq_text += f"\n{i+1}. {wrap_text(q['prompt'], 50)}\n"
        mcq_text += f"   Choices: {q['candidate_answers']}\n"
        mcq_text += f"   Ground Truth: {q['ground_truth_answer']}\n"

    ax_mcq.text(0.02, 0.98, mcq_text, transform=ax_mcq.transAxes, fontsize=8,
                verticalalignment="top", family="monospace")

    plt.suptitle("Query Subtype Comparison: Binary Relation vs Relation MCQ", fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def create_relation_axis_comparison(
    queries: List[Dict],
    labeled_image: Image.Image,
    output_path: Path,
) -> None:
    """Create visualization comparing all 4 relation axes."""

    axes_data = {
        "left_right": {"color": "#E74C3C", "desc": "Horizontal position (x-axis)"},
        "above_below": {"color": "#3498DB", "desc": "Vertical position (y-axis)"},
        "foreground_background": {"color": "#27AE60", "desc": "Depth from viewer (z-order)"},
        "front_behind": {"color": "#9B59B6", "desc": "Allocentric depth (oriented)"},
    }

    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.8, 1, 1], hspace=0.3, wspace=0.2)

    # Image
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(labeled_image)
    ax_img.set_title("Benchmark Image", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]

    for idx, (axis, info) in enumerate(axes_data.items()):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")
        ax.set_title(f"{prettify(axis)}\n({info['desc']})", fontsize=11, fontweight="bold", color=info["color"])

        axis_queries = [q for q in queries if q.get("relation_axis") == axis]

        text = f"Total queries: {len(axis_queries)}\n"
        text += f"Ground truth computation:\n"

        if axis == "left_right":
            text += "  dx = target.x - reference.x\n"
            text += "  return 'right' if dx > 0 else 'left'\n\n"
        elif axis == "above_below":
            text += "  dy = target.y - reference.y\n"
            text += "  return 'below' if dy > 0 else 'above'\n"
            text += "  (y increases downward in image coords)\n\n"
        elif axis == "foreground_background":
            text += "  return 'foreground' if target.z_order > ref.z_order\n"
            text += "  (higher z_order = closer to viewer)\n\n"
        elif axis == "front_behind":
            text += "  Project into oriented coordinate frame\n"
            text += "  forward_proj based on orientation\n"
            text += "  return 'in_front' if forward_proj > 0\n\n"

        text += "Examples:\n"
        for i, q in enumerate(axis_queries[:3]):
            text += f"{i+1}. {wrap_text(q['prompt'], 55)}\n"
            text += f"   GT: {q['ground_truth_answer']}\n\n"

        ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", family="monospace")

    plt.suptitle("Relation Axis Comparison", fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def create_orientation_comparison(
    queries: List[Dict],
    labeled_image: Image.Image,
    output_path: Path,
) -> None:
    """Create visualization comparing all 4 allocentric orientations."""

    orientations = ["camera", "away", "left", "right"]
    orientation_info = {
        "camera": {"forward": "(0, 1)", "right": "(-1, 0)", "desc": "Facing toward camera"},
        "away": {"forward": "(0, -1)", "right": "(1, 0)", "desc": "Facing away from camera"},
        "left": {"forward": "(-1, 0)", "right": "(0, -1)", "desc": "Facing left of image"},
        "right": {"forward": "(1, 0)", "right": "(0, 1)", "desc": "Facing right of image"},
    }

    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.8, 1, 1], hspace=0.3, wspace=0.2)

    # Image
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(labeled_image)
    ax_img.set_title("Benchmark Image", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    colors = ["#E74C3C", "#3498DB", "#27AE60", "#F39C12"]
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]

    for idx, orient in enumerate(orientations):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")

        info = orientation_info[orient]
        ax.set_title(f"Orientation: {prettify(orient)}\n({info['desc']})",
                     fontsize=11, fontweight="bold", color=colors[idx])

        orient_queries = [q for q in queries if q.get("orientation") == orient]

        text = f"Coordinate Transform:\n"
        text += f"  Forward vector: {info['forward']}\n"
        text += f"  Right vector: {info['right']}\n\n"
        text += f"Total queries: {len(orient_queries)}\n\n"
        text += "Examples:\n"

        for i, q in enumerate(orient_queries[:3]):
            text += f"{i+1}. {wrap_text(q['prompt'], 55)}\n"
            text += f"   Anchor: {q.get('anchor_object')} | GT: {q['ground_truth_answer']}\n\n"

        ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", family="monospace")

    plt.suptitle("Allocentric Orientation Comparison\n(Mental rotation required)", fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate detailed query visualizations")
    parser.add_argument("--queries", "-q", required=True, help="Path to queries.json")
    parser.add_argument("--graph", "-g", help="Path to spatial_graph.json")
    parser.add_argument("--image", "-i", help="Path to source image (auto-detected if not provided)")
    parser.add_argument("--output-dir", "-o", default="query_visualizations_detailed", help="Output directory")
    args = parser.parse_args()

    # Load queries
    queries_data = load_json(args.queries)
    queries = queries_data.get("queries", [])

    # Find image path
    image_path = args.image
    if not image_path:
        raw_path = queries_data.get("image_path", "")
        if raw_path:
            # Try various resolutions
            candidates = [
                Path(raw_path),
                Path(args.queries).parent.parent.parent / raw_path,
                Path.cwd() / raw_path,
            ]
            for c in candidates:
                if c.exists():
                    image_path = str(c)
                    break

    if not image_path or not Path(image_path).exists():
        print(f"Warning: Could not find image. Using placeholder.")
        # Create a placeholder image
        img = Image.new("RGB", (640, 480), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        draw.text((200, 240), "Image not found", fill=(100, 100, 100))
        labeled_image = img
    else:
        labeled_image = render_labeled_image(image_path, args.graph)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(queries)} queries")
    print(f"Output directory: {output_dir}")
    print()

    # Generate all visualizations
    print("Generating visualizations...")

    create_dimension_summary_table(queries, output_dir / "01_dimension_summary.png")
    create_task_type_comparison(queries, labeled_image, output_dir / "02_task_type_comparison.png")
    create_frame_type_comparison(queries, labeled_image, output_dir / "03_frame_type_comparison.png")
    create_query_subtype_comparison(queries, labeled_image, output_dir / "04_query_subtype_comparison.png")
    create_relation_axis_comparison(queries, labeled_image, output_dir / "05_relation_axis_comparison.png")
    create_orientation_comparison(queries, labeled_image, output_dir / "06_orientation_comparison.png")

    print(f"\nDone! Generated 6 visualization files in {output_dir}/")


if __name__ == "__main__":
    main()
