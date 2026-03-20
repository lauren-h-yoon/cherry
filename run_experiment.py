#!/usr/bin/env python3
"""
run_experiment.py — Batch experiment runner for VLM spatial understanding evaluation.

Runs systematic experiments across multiple images and model providers,
aggregates metrics, and produces a visual HTML report.

Examples
--------
    # Run Claude on all photos
    python run_experiment.py --images photos/*.jpg --providers claude

    # Compare Claude vs GPT-4o on 10 images
    python run_experiment.py \\
        --images photos/*.jpg \\
        --providers claude openai \\
        --max-images 10 \\
        --output-dir experiments/exp_001

    # With pre-computed spatial graphs (ground truth)
    python run_experiment.py \\
        --images photos/*.jpg \\
        --graphs spatial_outputs/ \\
        --providers claude openai \\
        --output-dir experiments/exp_002

    # Qwen via vLLM server
    python run_experiment.py \\
        --images photos/*.jpg \\
        --providers vllm \\
        --vllm-url http://localhost:8000/v1 \\
        --model Qwen/Qwen2-VL-7B-Instruct

Output
------
    experiments/exp_001/
    ├── results.json          ← aggregated metrics across all images + providers
    ├── per_image/            ← one JSON per (image, provider) pair
    │   ├── kitchen_claude.json
    │   └── kitchen_openai.json
    └── analysis.html         ← visual report comparing providers

Unity requirement
-----------------
Unity must be running with CherryUnityBridge.cs attached before starting.
The scene is reset between runs automatically.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from unity_bridge import UnityBridge
from spatial_agent.model_providers import create_model_provider
from spatial_graph_to_unity import convert_for_evaluation
from run_unity_eval import (
    SYSTEM_PROMPT,
    run_agentic_placement,
    evaluate_placement,
)


# ─── Experiment runner ────────────────────────────────────────────────────────

def find_graph_for_image(image_path: Path, graphs_dir: Optional[Path]) -> Optional[Dict]:
    """
    Look for a spatial graph JSON that corresponds to image_path.

    Checks for:
      {graphs_dir}/{image_stem}_spatial_graph.json
      {graphs_dir}/{image_stem}.json
      {image_path.parent}/{image_stem}_spatial_graph.json
    """
    if graphs_dir is None:
        return None

    candidates = [
        graphs_dir / f"{image_path.stem}_spatial_graph.json",
        graphs_dir / f"{image_path.stem}.json",
        image_path.parent / f"{image_path.stem}_spatial_graph.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate) as f:
                raw = json.load(f)
            # Normalise nodes → entities
            if "nodes" in raw and "entities" not in raw:
                return convert_for_evaluation(raw)
            return raw
    return None


def run_single(
    image_path: Path,
    provider_name: str,
    model_name: Optional[str],
    bridge: UnityBridge,
    graph_data: Optional[Dict],
    verbose: bool,
    provider_kwargs: Dict,
) -> Dict:
    """
    Run one (image, provider) experiment and return the result dict.
    """
    provider = create_model_provider(provider_name, model_name=model_name, **provider_kwargs)

    # Reset scene
    bridge.initialize_scene()

    t0 = time.time()
    try:
        final_state = run_agentic_placement(
            image_path=str(image_path),
            provider=provider,
            bridge=bridge,
            graph_data=graph_data,
            verbose=verbose,
        )
        elapsed = round(time.time() - t0, 1)
        eval_results = evaluate_placement(final_state, graph_data)
        error = None
    except Exception as exc:
        elapsed = round(time.time() - t0, 1)
        final_state = None
        eval_results = {}
        error = str(exc)
        print(f"  ERROR: {exc}", file=sys.stderr)

    placed_objects = []
    if final_state:
        placed_objects = [
            {"id": o.id, "label": o.label,
             "x": o.x, "y": o.y, "z": o.z,
             "color": o.color, "scale": o.scale}
            for o in final_state.objects
        ]

    return {
        "image": str(image_path),
        "image_stem": image_path.stem,
        "provider": provider_name,
        "model": provider.model_name,
        "has_ground_truth": graph_data is not None,
        "elapsed_s": elapsed,
        "placed_objects": placed_objects,
        "evaluation": eval_results,
        "error": error,
    }


def aggregate_results(all_results: List[Dict]) -> Dict:
    """
    Compute per-provider aggregate metrics across all images.
    """
    from collections import defaultdict

    by_provider: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        if r["error"] is None:
            by_provider[r["provider"]].append(r)

    aggregated = {}
    for provider, results in by_provider.items():
        n = len(results)
        if n == 0:
            continue

        def _avg(key):
            vals = [r["evaluation"].get(key) for r in results if r["evaluation"].get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        aggregated[provider] = {
            "model": results[0]["model"],
            "n_images": n,
            "avg_objects_placed": round(
                sum(r["evaluation"].get("objects_placed", 0) for r in results) / n, 1
            ),
            "avg_coverage": _avg("coverage"),
            "avg_left_right_accuracy": _avg("left_right_accuracy"),
            "avg_near_far_accuracy": _avg("near_far_accuracy"),
            "avg_elapsed_s": round(sum(r["elapsed_s"] for r in results) / n, 1),
            "n_errors": sum(1 for r in all_results if r["provider"] == provider and r["error"]),
        }

    return aggregated


# ─── HTML report ──────────────────────────────────────────────────────────────

def _metric_cell(value, is_better_high: bool = True) -> str:
    """Return a coloured <td> for a metric value."""
    if value is None:
        return "<td>—</td>"
    pct = f"{value * 100:.1f}%" if isinstance(value, float) and value <= 1 else str(value)
    if isinstance(value, float) and value <= 1:
        if value >= 0.8:
            color = "#c6efce"
        elif value >= 0.6:
            color = "#ffeb9c"
        else:
            color = "#ffc7ce"
    else:
        color = "transparent"
    return f'<td style="background:{color}">{pct}</td>'


def generate_html_report(
    all_results: List[Dict],
    aggregated: Dict,
    output_path: Path,
    experiment_name: str,
) -> None:
    """Write a self-contained HTML analysis report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    providers = sorted(aggregated.keys())

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = ""
    for p in providers:
        a = aggregated[p]
        summary_rows += f"""
        <tr>
          <td><b>{p}</b></td>
          <td>{a['model']}</td>
          <td>{a['n_images']}</td>
          <td>{a['avg_objects_placed']}</td>
          {_metric_cell(a['avg_coverage'])}
          {_metric_cell(a['avg_left_right_accuracy'])}
          {_metric_cell(a['avg_near_far_accuracy'])}
          <td>{a['avg_elapsed_s']}s</td>
          <td>{a['n_errors']}</td>
        </tr>"""

    # ── Per-image table ────────────────────────────────────────────────────────
    images = sorted({r["image_stem"] for r in all_results})
    detail_rows = ""
    for img in images:
        img_results = [r for r in all_results if r["image_stem"] == img]
        first = True
        for r in sorted(img_results, key=lambda x: x["provider"]):
            ev = r["evaluation"]
            img_cell = f'<td rowspan="{len(img_results)}">{img}</td>' if first else ""
            first = False
            err = f'<span style="color:red">{r["error"]}</span>' if r["error"] else "✓"
            detail_rows += f"""
        <tr>
          {img_cell}
          <td>{r['provider']}</td>
          <td>{ev.get('objects_placed', '—')}</td>
          {_metric_cell(ev.get('coverage'))}
          {_metric_cell(ev.get('left_right_accuracy'))}
          {_metric_cell(ev.get('near_far_accuracy'))}
          <td>{r['elapsed_s']}s</td>
          <td>{err}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Cherry Experiment: {experiment_name}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
    h1 {{ color: #333; }}
    h2 {{ color: #555; margin-top: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #f5f5f5; font-weight: 600; }}
    tr:hover {{ background: #fafafa; }}
    .meta {{ color: #888; font-size: 0.9rem; margin-bottom: 1rem; }}
  </style>
</head>
<body>
  <h1>Cherry: VLM Spatial Understanding Experiment</h1>
  <p class="meta">Generated: {timestamp} &nbsp;|&nbsp; Experiment: {experiment_name}</p>

  <h2>Provider Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Provider</th><th>Model</th><th>Images</th>
        <th>Avg Objects</th><th>Coverage</th>
        <th>Left/Right Acc</th><th>Near/Far Acc</th>
        <th>Avg Time</th><th>Errors</th>
      </tr>
    </thead>
    <tbody>{summary_rows}</tbody>
  </table>

  <h2>Per-Image Results</h2>
  <table>
    <thead>
      <tr>
        <th>Image</th><th>Provider</th><th>Objects</th>
        <th>Coverage</th><th>Left/Right</th><th>Near/Far</th>
        <th>Time</th><th>Status</th>
      </tr>
    </thead>
    <tbody>{detail_rows}</tbody>
  </table>

  <h2>Notes</h2>
  <ul>
    <li><b>Coverage</b>: fraction of ground-truth entities placed by the model.</li>
    <li><b>Left/Right Accuracy</b>: fraction of pairwise L/R relations matching ground truth.</li>
    <li><b>Near/Far Accuracy</b>: fraction of pairwise depth relations matching ground truth.</li>
    <li>— means no ground-truth spatial graph was available for that image.</li>
    <li>Cells coloured: green ≥ 80%, yellow ≥ 60%, red &lt; 60%.</li>
  </ul>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"[Experiment] HTML report saved to: {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cherry batch experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Inputs
    parser.add_argument("--images", "-i", nargs="+", required=True,
                        help="Image file paths (supports glob, e.g. photos/*.jpg)")
    parser.add_argument("--graphs", "-g",
                        help="Directory containing spatial graph JSON files "
                             "(matched by image stem, e.g. kitchen_spatial_graph.json)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit to first N images (useful for quick tests)")

    # Models
    parser.add_argument("--providers", "-p", nargs="+",
                        choices=["claude", "openai", "vllm", "qwen", "ollama"],
                        default=["claude"],
                        help="VLM providers to evaluate (default: claude)")
    parser.add_argument("--model", "-m",
                        help="Model name override (applied to all providers)")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1",
                        help="vLLM server URL (default: http://localhost:8000/v1)")

    # Unity
    parser.add_argument("--unity-port", type=int, default=5555,
                        help="Unity bridge port (default: 5555)")
    parser.add_argument("--unity-url",
                        help="Full Unity bridge URL (overrides --unity-port)")
    parser.add_argument("--unity-timeout", type=float, default=30.0,
                        help="Seconds to wait for Unity (default: 30)")

    # Eval
    parser.add_argument("--output-dir", "-o", default="experiments/exp_001",
                        help="Output directory (default: experiments/exp_001)")
    parser.add_argument("--name",
                        help="Experiment name (defaults to output directory name)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress verbose per-turn output")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (image, provider) pairs that already have output files")

    args = parser.parse_args()

    # ── Resolve image list ────────────────────────────────────────────────────
    import glob as _glob
    image_paths: List[Path] = []
    for pattern in args.images:
        expanded = _glob.glob(pattern)
        if expanded:
            image_paths.extend(Path(p) for p in expanded)
        else:
            p = Path(pattern)
            if p.exists():
                image_paths.append(p)
            else:
                print(f"WARNING: no files matched: {pattern}", file=sys.stderr)

    image_paths = sorted(set(image_paths))
    if not image_paths:
        print("ERROR: no images found.", file=sys.stderr)
        sys.exit(1)

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    graphs_dir = Path(args.graphs) if args.graphs else None
    out_dir = Path(args.output_dir)
    per_image_dir = out_dir / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = args.name or out_dir.name
    verbose = not args.quiet

    print(f"[Experiment] {experiment_name}")
    print(f"[Experiment] Images: {len(image_paths)}  |  Providers: {args.providers}")
    print(f"[Experiment] Output: {out_dir}")

    # ── Connect to Unity ──────────────────────────────────────────────────────
    unity_url = args.unity_url or f"http://localhost:{args.unity_port}"
    bridge = UnityBridge(base_url=unity_url)
    print(f"\n[Experiment] Connecting to Unity at {unity_url}…")
    bridge.wait_for_unity(timeout_s=args.unity_timeout)

    # ── Provider kwargs ───────────────────────────────────────────────────────
    all_results: List[Dict] = []
    total = len(image_paths) * len(args.providers)
    done = 0

    for image_path in image_paths:
        graph_data = find_graph_for_image(image_path, graphs_dir)
        if graph_data:
            n_entities = len(graph_data.get("entities", []))
            print(f"\n[Image] {image_path.name}  (ground truth: {n_entities} entities)")
        else:
            print(f"\n[Image] {image_path.name}  (no ground truth)")

        for provider_name in args.providers:
            done += 1
            out_file = per_image_dir / f"{image_path.stem}_{provider_name}.json"

            if args.skip_existing and out_file.exists():
                print(f"  [{done}/{total}] Skipping {provider_name} (already done)")
                with open(out_file) as f:
                    all_results.append(json.load(f))
                continue

            print(f"  [{done}/{total}] Running {provider_name}…")

            provider_kwargs = {}
            if provider_name in ("vllm", "qwen"):
                provider_kwargs["base_url"] = args.vllm_url

            result = run_single(
                image_path=image_path,
                provider_name=provider_name,
                model_name=args.model,
                bridge=bridge,
                graph_data=graph_data,
                verbose=verbose,
                provider_kwargs=provider_kwargs,
            )
            all_results.append(result)

            # Save per-image result
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"       saved → {out_file.name}")

            # Brief per-run summary
            ev = result["evaluation"]
            if result["error"]:
                print(f"       ERROR: {result['error']}")
            else:
                cov = ev.get("coverage")
                lr = ev.get("left_right_accuracy")
                nf = ev.get("near_far_accuracy")
                objs = ev.get("objects_placed", 0)
                cov_s = f"{cov:.0%}" if cov is not None else "—"
                lr_s  = f"{lr:.0%}"  if lr  is not None else "—"
                nf_s  = f"{nf:.0%}"  if nf  is not None else "—"
                print(f"       objects={objs}  coverage={cov_s}  "
                      f"L/R={lr_s}  N/F={nf_s}  ({result['elapsed_s']}s)")

    # ── Aggregate & save ──────────────────────────────────────────────────────
    aggregated = aggregate_results(all_results)

    print("\n" + "═" * 60)
    print("[Experiment] Aggregate results:")
    for provider, agg in aggregated.items():
        print(f"  {provider} ({agg['model']}):")
        print(f"    images={agg['n_images']}  "
              f"avg_objects={agg['avg_objects_placed']}  "
              f"coverage={agg['avg_coverage']}  "
              f"L/R={agg['avg_left_right_accuracy']}  "
              f"N/F={agg['avg_near_far_accuracy']}")

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "experiment": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "providers": args.providers,
            "n_images": len(image_paths),
            "max_turns": args.max_turns,
            "aggregated": aggregated,
            "all_results": all_results,
        }, f, indent=2)
    print(f"\n[Experiment] Results saved to: {results_path}")

    html_path = out_dir / "analysis.html"
    generate_html_report(all_results, aggregated, html_path, experiment_name)


if __name__ == "__main__":
    main()
