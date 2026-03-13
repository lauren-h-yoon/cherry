#!/usr/bin/env python3
"""
analyze_query_results.py - Summarize query benchmark evaluation results.

Reads `evaluation_results.json` from `run_query_benchmark.py` and:
- prints a readable summary to stdout
- saves a text report
- saves a machine-readable summary JSON
- saves a few bar-chart PNGs for quick inspection

Example:
    python analyze_query_results.py \
        --input query_benchmark_outputs/evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def update_bucket(buckets: Dict[str, Dict[str, float]], key: str, correct: bool) -> None:
    bucket = buckets.setdefault(key, {"count": 0, "correct": 0})
    bucket["count"] += 1
    bucket["correct"] += int(correct)


def finalize_buckets(buckets: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    for bucket in buckets.values():
        bucket["accuracy"] = bucket["correct"] / bucket["count"] if bucket["count"] else 0.0
    return dict(sorted(buckets.items(), key=lambda item: item[0]))


def format_bucket_table(title: str, buckets: Dict[str, Dict[str, float]]) -> List[str]:
    lines = ["", title, "-" * len(title)]
    for key, stats in buckets.items():
        lines.append(f"{key}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['count']})")
    return lines


def prettify_label(label: str) -> str:
    mapping = {
        "allocentric_qa": "Allocentric QA",
        "egocentric_qa": "Egocentric QA",
        "viewer_centered": "Viewer-centered",
        "object_to_object": "Object-to-object",
        "binary_relation": "Binary relation",
        "object_retrieval": "Object retrieval",
        "relation_mcq": "Relation MCQ",
        "above_below": "Above / below",
        "foreground_background": "Foreground / background",
        "front_behind": "Front / behind",
        "left_right": "Left / right",
        "ego_v1": "Ego v1",
        "ego_v2": "Ego v2",
        "allo_v1": "Allo v1",
        "allo_v2": "Allo v2",
    }
    if label in mapping:
        return mapping[label]
    return label.replace("_", " ").title()


def format_task_frame_subtype_table(rows: List[Dict[str, object]]) -> List[str]:
    headers = ["Task", "Frame", "Subtype", "Accuracy", "Correct", "Count"]
    formatted_rows = [
        [
            prettify_label(str(row["task_type"])),
            prettify_label(str(row["frame_type"])),
            prettify_label(str(row["query_subtype"])),
            f"{float(row['accuracy']):.1%}",
            str(row["correct"]),
            str(row["count"]),
        ]
        for row in rows
    ]

    widths = []
    for col_idx, header in enumerate(headers):
        cell_width = max([len(header)] + [len(row[col_idx]) for row in formatted_rows])
        widths.append(cell_width)

    def format_row(values: List[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    lines = ["", "By Task / Frame / Subtype", "-" * len("By Task / Frame / Subtype")]
    lines.append(format_row(headers))
    lines.append("-+-".join("-" * width for width in widths))
    for row in formatted_rows:
        lines.append(format_row(row))
    return lines


def task_frame_subtype_note() -> List[str]:
    bucket_cap = 4
    return [
        "",
        "Note",
        "----",
        "Counts in the Task / Frame / Subtype table aggregate over relation-axis buckets.",
        f"Generation is capped at {bucket_cap} queries per (task_type, frame_type, query_subtype, relation_axis) bucket before these rows are combined.",
        "Legal relation axes are:",
        "- Egocentric QA + Viewer-centered: Left / right, Above / below, Foreground / background",
        "- Egocentric QA + Object-to-object: Left / right, Above / below, Front / behind",
        "- Allocentric QA + Viewer-centered: Left / right, Front / behind",
        "- Allocentric QA + Object-to-object: Left / right, Front / behind",
        "Some rows fall below the nominal total because invalid or ambiguous queries are filtered out before evaluation.",
    ]


def analyze(results: Dict) -> Dict:
    rows: List[Dict] = results.get("results", [])

    by_task_type: Dict[str, Dict[str, float]] = {}
    by_frame_type: Dict[str, Dict[str, float]] = {}
    by_query_subtype: Dict[str, Dict[str, float]] = {}
    by_relation_axis: Dict[str, Dict[str, float]] = {}
    by_template_id: Dict[str, Dict[str, float]] = {}
    by_task_frame_subtype: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    by_task_template: Dict[str, Dict[str, float]] = {}
    mistakes: Counter[Tuple[str, str]] = Counter()

    for row in rows:
        correct = bool(row.get("correct", False))
        update_bucket(by_task_type, row.get("task_type", "unknown"), correct)
        update_bucket(by_frame_type, row.get("frame_type", "unknown"), correct)
        update_bucket(by_query_subtype, row.get("query_subtype", "unknown"), correct)
        update_bucket(by_relation_axis, row.get("relation_axis", "unknown"), correct)
        update_bucket(by_template_id, row.get("template_id", "unknown"), correct)

        combo = (
            row.get("task_type", "unknown"),
            row.get("frame_type", "unknown"),
            row.get("query_subtype", "unknown"),
        )
        update_bucket(by_task_frame_subtype, combo, correct)
        task_template = f"{row.get('task_type', 'unknown')} | {row.get('template_id', 'unknown')}"
        update_bucket(by_task_template, task_template, correct)

        if not correct:
            mistakes[(row.get("ground_truth_answer", ""), row.get("normalized_model_answer", ""))] += 1

    task_frame_subtype_rows = []
    for (task_type, frame_type, query_subtype), stats in sorted(by_task_frame_subtype.items()):
        accuracy = stats["correct"] / stats["count"] if stats["count"] else 0.0
        task_frame_subtype_rows.append(
            {
                "task_type": task_type,
                "frame_type": frame_type,
                "query_subtype": query_subtype,
                "correct": stats["correct"],
                "count": stats["count"],
                "accuracy": accuracy,
            }
        )

    return {
        "overall": results.get("overall", {}),
        "model": results.get("model", {}),
        "by_task_type": finalize_buckets(by_task_type),
        "by_frame_type": finalize_buckets(by_frame_type),
        "by_query_subtype": finalize_buckets(by_query_subtype),
        "by_relation_axis": finalize_buckets(by_relation_axis),
        "by_template_id": finalize_buckets(by_template_id),
        "by_task_frame_subtype": {
            f"{row['task_type']} | {row['frame_type']} | {row['query_subtype']}": {
                "correct": row["correct"],
                "count": row["count"],
                "accuracy": row["accuracy"],
            }
            for row in task_frame_subtype_rows
        },
        "task_frame_subtype_rows": task_frame_subtype_rows,
        "by_task_template": finalize_buckets(by_task_template),
        "top_mistakes": [
            {"ground_truth": gt, "predicted": pred, "count": count}
            for (gt, pred), count in mistakes.most_common(10)
        ],
    }


def render_summary_text(summary: Dict, input_path: Path) -> str:
    model = summary.get("model", {})
    overall = summary.get("overall", {})

    lines = [
        "QUERY BENCHMARK ANALYSIS",
        "=" * 80,
        f"Input: {input_path.resolve()}",
        f"Model: {model.get('model_name', 'unknown')} ({model.get('provider', 'unknown')})",
        f"Overall accuracy: {overall.get('accuracy', 0.0):.1%} ({overall.get('correct', 0)}/{overall.get('count', 0)})",
    ]

    lines.extend(format_bucket_table("By Task Type", summary["by_task_type"]))
    lines.extend(format_bucket_table("By Frame Type", summary["by_frame_type"]))
    lines.extend(format_bucket_table("By Query Subtype", summary["by_query_subtype"]))
    lines.extend(format_bucket_table("By Relation Axis", summary["by_relation_axis"]))
    lines.extend(format_bucket_table("By Template ID", summary["by_template_id"]))
    lines.extend(format_task_frame_subtype_table(summary["task_frame_subtype_rows"]))
    lines.extend(task_frame_subtype_note())
    lines.extend(format_bucket_table("By Task / Template", summary["by_task_template"]))

    lines.extend(["", "Top Mistakes", "-----------"])
    if not summary["top_mistakes"]:
        lines.append("No mistakes recorded.")
    else:
        for item in summary["top_mistakes"]:
            pred_display = item["predicted"] if item["predicted"] else "<empty>"
            lines.append(f"GT={item['ground_truth']} | Pred={pred_display}: {item['count']}")

    return "\n".join(lines)


def save_chart(
    title: str,
    buckets: Dict[str, Dict[str, float]],
    output_path: Path,
    figsize: Tuple[float, float] = (8.0, 4.5),
) -> None:
    if not buckets:
        return

    labels = list(buckets.keys())
    accuracies = [buckets[label]["accuracy"] * 100.0 for label in labels]
    pretty_labels = [prettify_label(label) for label in labels]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(labels)), accuracies, color="#3F6C8C", edgecolor="#234257", linewidth=1.0)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(pretty_labels, rotation=25, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D7E0E8", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            accuracy + 1.0,
            f"{accuracy:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1E2F3C",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_task_frame_subtype_table(rows: List[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        return

    bucket_cap = 4
    headers = ["Task", "Frame", "Subtype", "Accuracy", "Correct", "Count"]
    cell_text = [
        [
            prettify_label(str(row["task_type"])),
            prettify_label(str(row["frame_type"])),
            prettify_label(str(row["query_subtype"])),
            f"{float(row['accuracy']):.1%}",
            str(row["correct"]),
            str(row["count"]),
        ]
        for row in rows
    ]

    plt.style.use("seaborn-v0_8-white")
    fig_height = max(5.8, 0.48 * (len(cell_text) + 3))
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    ax.axis("off")
    ax.set_title("Task / Frame / Subtype Breakdown", fontsize=15, weight="bold", pad=14)

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.14, 1.0, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#C9D5DF")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#3F6C8C")
            cell.set_text_props(color="white", weight="bold")
        elif row_idx % 2 == 1:
            cell.set_facecolor("#F4F8FB")
        else:
            cell.set_facecolor("#FFFFFF")

        if col_idx in (0, 1, 2):
            cell.get_text().set_ha("left")

    note = (
        f"Counts aggregate over relation-axis buckets capped at {bucket_cap} queries each.\n"
        "Axes: Ego viewer = Left/right, Above/below, Foreground/background; "
        "Ego object = Left/right, Above/below, Front/behind; "
        "Allo viewer/object = Left/right, Front/behind.\n"
        "Some totals fall below the nominal cap because invalid or ambiguous queries are filtered out."
    )
    ax.text(
        0.0,
        0.03,
        note,
        transform=ax.transAxes,
        fontsize=9,
        color="#324B5C",
        va="bottom",
        ha="left",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_outputs(summary: Dict, input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_text = render_summary_text(summary, input_path)
    (output_dir / "analysis_summary.txt").write_text(summary_text)
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    save_chart("Accuracy by Task Type", summary["by_task_type"], output_dir / "accuracy_by_task_type.png")
    save_chart("Accuracy by Frame Type", summary["by_frame_type"], output_dir / "accuracy_by_frame_type.png")
    save_chart("Accuracy by Query Subtype", summary["by_query_subtype"], output_dir / "accuracy_by_query_subtype.png")
    save_chart("Accuracy by Relation Axis", summary["by_relation_axis"], output_dir / "accuracy_by_relation_axis.png")
    save_chart("Accuracy by Template ID", summary["by_template_id"], output_dir / "accuracy_by_template_id.png")
    save_task_frame_subtype_table(summary["task_frame_subtype_rows"], output_dir / "task_frame_subtype_table.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze query benchmark evaluation results")
    parser.add_argument("--input", "-i", required=True, help="Path to evaluation_results.json")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory for saved analysis outputs. Defaults to the input file's parent directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    results = load_results(args.input)
    summary = analyze(results)
    summary_text = render_summary_text(summary, input_path)

    print(summary_text)
    save_outputs(summary, input_path, output_dir)
    print(f"\nSaved analysis outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
