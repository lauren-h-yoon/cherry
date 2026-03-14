#!/usr/bin/env python3
"""
run_query_benchmark.py - Generate and optionally evaluate query-based spatial benchmarks.

This script is the entry point for the query benchmark:

1. Load a Stage 1 `spatial_graph.json`
   - produced by the perception pipeline (SAM3 + depth)
2. Generate spatial reasoning queries
   - egocentric QA
   - allocentric QA
3. Optionally render a labeled benchmark image
   - with object boxes and labels drawn on top
4. Optionally evaluate a model
   - exact-match on returned label only

Outputs:
- `<output-dir>/<image_name>__<provider>__<model>/queries.json`
- `<output-dir>/<image_name>__<provider>__<model>/queries_readable.txt`
- `<output-dir>/<image_name>__<provider>__<model>/benchmark_image.png` (if rendered locally)
- `<output-dir>/<image_name>__<provider>__<model>/evaluation_results.json` (if `--evaluate`)

Examples:
    # Generate queries only
    python run_query_benchmark.py \
        --graph spatial_outputs/living_room2_spatial_graph.json \
        --output-dir query_benchmark_outputs/

    # Generate queries and render labeled benchmark image
    python run_query_benchmark.py \
        --graph spatial_outputs/living_room2_spatial_graph.json \
        --render-image \
        --output-dir query_benchmark_outputs/

    # Generate queries and evaluate with OpenAI
    python run_query_benchmark.py \
        --graph spatial_outputs/living_room2_spatial_graph.json \
        --render-image \
        --evaluate \
        --provider openai \
        --model gpt-4o \
        --output-dir query_benchmark_outputs/
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from query_benchmark.generator import generate_queries
from query_benchmark.ground_truth import (
    load_graph,
    render_labeled_image,
    resolve_image_path,
    resolve_stage1_labeled_image,
)


def _load_model_providers():
    module_path = Path(__file__).parent / "spatial_agent" / "model_providers.py"
    spec = importlib.util.spec_from_file_location("query_benchmark_model_providers", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model providers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUERY_SYSTEM_PROMPT = """You are answering spatial reasoning questions about an image with labeled bounding boxes.

Use the labels shown in the image exactly.
Return only the answer.
Do not explain your reasoning.
If the valid answer choices are shown in the prompt, output exactly one of them and nothing else.
"""


def _basic_normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().replace("_", " ").strip(".").split())


def normalize_answer(text: str, candidate_answers: List[str] | None = None) -> str:
    normalized = _basic_normalize(text)
    if not candidate_answers:
        return normalized.replace(" ", "_")

    candidate_map = {_basic_normalize(answer): answer.lower() for answer in candidate_answers}
    if normalized in candidate_map:
        return candidate_map[normalized]

    for normalized_candidate, original_candidate in sorted(candidate_map.items(), key=lambda item: len(item[0]), reverse=True):
        if normalized_candidate and normalized_candidate in normalized:
            return original_candidate

    return normalized.replace(" ", "_")


def is_retriable_error(exc: Exception) -> bool:
    name = exc.__class__.__name__
    if name == "RateLimitError":
        return True
    text = str(exc).lower()
    if "rate limit" in text or "429" in text:
        return True
    if "could not parse the json body of your request" in text:
        return True
    return False


def sanitize_name(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("_")
    while "__" in result:
        result = result.replace("__", "_")
    return result or "unknown"


def graph_image_name(graph_path: str) -> str:
    stem = Path(graph_path).stem
    if stem.endswith("_spatial_graph"):
        stem = stem[: -len("_spatial_graph")]
    return sanitize_name(stem)


def resolve_run_output_dir(base_output_dir: Path, graph_path: str, provider: str, model_name: str | None, evaluate: bool) -> Path:
    image_name = graph_image_name(graph_path)
    if evaluate:
        model_part = sanitize_name(model_name or provider)
        return base_output_dir / f"{image_name}__{sanitize_name(provider)}__{model_part}"
    return base_output_dir / f"{image_name}__queries"


def write_readable_queries(output_path: Path, generated: Dict) -> None:
    lines: List[str] = []
    metadata = generated.get("metadata", {})
    queries = generated.get("queries", [])

    lines.append("QUERY BENCHMARK SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Graph: {metadata.get('graph_path', 'unknown')}")
    lines.append(f"Image: {generated.get('image_path', 'unknown')}")
    lines.append(f"Image size: {generated.get('image_size', [])}")
    lines.append(f"Entities: {metadata.get('num_entities', '?')}")
    lines.append(f"Queries: {metadata.get('num_queries', len(queries))}")
    lines.append("")

    current_group = None
    for query in queries:
        group = (query["task_type"], query["frame_type"], query["query_subtype"])
        if group != current_group:
            current_group = group
            lines.append("-" * 80)
            lines.append(
                f"{query['task_type']} | {query['frame_type']} | {query['query_subtype']}"
            )
            lines.append("-" * 80)

        lines.append(f"[{query['query_id']}]")
        lines.append(f"Prompt: {query['prompt']}")
        if query.get("anchor_object"):
            lines.append(f"Anchor: {query['anchor_object']}")
        if query.get("reference_object"):
            lines.append(f"Reference: {query['reference_object']}")
        if query.get("target_object"):
            lines.append(f"Target: {query['target_object']}")
        if query.get("orientation"):
            lines.append(f"Orientation: {query['orientation']}")
        lines.append(f"Relation axis: {query['relation_axis']}")
        lines.append(f"Candidate answers: {', '.join(query['candidate_answers'])}")
        lines.append(f"Ground truth: {query['ground_truth_answer']}")
        if query.get("metadata"):
            lines.append(f"Metadata: {json.dumps(query['metadata'], sort_keys=True)}")
        lines.append("")

    output_path.write_text("\n".join(lines))


def evaluate_queries(
    image_path: str,
    queries: List[Dict],
    provider: str,
    model_name: str | None,
    partial_output_path: Path | None = None,
    max_retries: int = 6,
    **provider_kwargs,
) -> Dict:
    providers_module = _load_model_providers()
    create_model_provider = providers_module.create_model_provider
    provider_client = create_model_provider(provider, model_name=model_name, **provider_kwargs)
    results = []
    correct = 0
    total = len(queries)

    for idx, query in enumerate(queries, start=1):
        print(
            f"Evaluating {idx}/{total}: "
            f"{query['task_type']} | {query['frame_type']} | {query['query_subtype']} | {query['query_id']}",
            flush=True,
        )
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = provider_client.generate(
                    image_path=image_path,
                    prompt=query["prompt"],
                    system_prompt=QUERY_SYSTEM_PROMPT,
                    tools=None,
                )
                break
            except Exception as exc:
                if not is_retriable_error(exc) or attempt == max_retries:
                    if partial_output_path is not None:
                        partial = {
                            "model": {"provider": provider, "model_name": provider_client.model_name},
                            "overall": {
                                "count": len(results),
                                "correct": correct,
                                "accuracy": correct / len(results) if results else 0.0,
                                "partial": True,
                            },
                            "results": results,
                        }
                        partial_output_path.write_text(json.dumps(partial, indent=2))
                    raise

                wait_seconds = min(8.0, 0.75 * (2 ** attempt))
                print(
                    f"  Retriable API error on {query['query_id']} (attempt {attempt + 1}/{max_retries + 1}): {exc.__class__.__name__}. "
                    f"Retrying in {wait_seconds:.2f}s...",
                    flush=True,
                )
                time.sleep(wait_seconds)

        if response is None:
            raise RuntimeError(f"Failed to get response for {query['query_id']}")
        model_answer = normalize_answer(response.text, query.get("candidate_answers"))
        gt = normalize_answer(query["ground_truth_answer"], query.get("candidate_answers"))
        is_correct = model_answer == gt
        if is_correct:
            correct += 1
        results.append(
            {
                "query_id": query["query_id"],
                "task_type": query["task_type"],
                "frame_type": query["frame_type"],
                "query_subtype": query["query_subtype"],
                "template_id": query.get("template_id"),
                "relation_axis": query.get("relation_axis"),
                "prompt": query["prompt"],
                "ground_truth_answer": query["ground_truth_answer"],
                "model_answer": response.text,
                "normalized_model_answer": model_answer,
                "correct": is_correct,
            }
        )

        if partial_output_path is not None:
            partial = {
                "model": {"provider": provider, "model_name": provider_client.model_name},
                "overall": {
                    "count": len(results),
                    "correct": correct,
                    "accuracy": correct / len(results) if results else 0.0,
                    "partial": True,
                },
                "results": results,
            }
            partial_output_path.write_text(json.dumps(partial, indent=2))

    by_type: Dict[str, Dict[str, float]] = {}
    for row in results:
        bucket = by_type.setdefault(row["task_type"], {"count": 0, "correct": 0})
        bucket["count"] += 1
        bucket["correct"] += int(row["correct"])
    for bucket in by_type.values():
        bucket["accuracy"] = bucket["correct"] / bucket["count"] if bucket["count"] else 0.0

    return {
        "model": {"provider": provider, "model_name": provider_client.model_name},
        "overall": {
            "count": len(results),
            "correct": correct,
            "accuracy": correct / len(results) if results else 0.0,
            "partial": False,
        },
        "by_task_type": by_type,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and optionally evaluate query benchmark prompts")
    parser.add_argument("--graph", "-g", required=True, help="Path to Stage 1 spatial_graph.json")
    parser.add_argument("--output-dir", "-o", default="query_benchmark_outputs", help="Base output directory")
    parser.add_argument("--queries-per-bucket", type=int, default=4, help="Maximum number of queries to keep per category bucket (default: 4)")
    parser.add_argument("--seed", type=int, help="Random seed for per-bucket query sampling")
    parser.add_argument("--max-queries", type=int, help="Optional cap on generated queries")
    parser.add_argument("--render-image", action="store_true", help="Render labeled benchmark image")
    parser.add_argument("--evaluate", action="store_true", help="Run a model and exact-match evaluate answers")
    parser.add_argument("--provider", choices=["claude", "qwen", "vllm", "openai", "ollama"], default="openai")
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        providers_module = _load_model_providers()
        print(json.dumps(providers_module.list_available_providers(), indent=2))
        return

    output_dir = resolve_run_output_dir(Path(args.output_dir), args.graph, args.provider, args.model, args.evaluate)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = generate_queries(
        args.graph,
        max_queries=args.max_queries,
        queries_per_bucket=args.queries_per_bucket,
        seed=args.seed,
    )
    queries_path = output_dir / "queries.json"
    readable_path = output_dir / "queries_readable.txt"
    with open(queries_path, "w") as f:
        json.dump(generated, f, indent=2)
    write_readable_queries(readable_path, generated)

    graph_data = load_graph(args.graph)
    source_image = resolve_image_path(graph_data, args.graph)
    benchmark_image = str(source_image)
    if args.render_image or args.evaluate:
        existing_labeled = resolve_stage1_labeled_image(args.graph)
        if existing_labeled is not None:
            benchmark_image = str(existing_labeled)
        else:
            benchmark_image = render_labeled_image(
                args.graph,
                str(output_dir / "benchmark_image.png"),
            )

    print(f"Generated {generated['metadata']['num_queries']} queries")
    print(f"Queries saved to: {queries_path}")
    print(f"Readable queries: {readable_path}")
    if benchmark_image != str(source_image):
        print(f"Benchmark image: {benchmark_image}")

    if not args.evaluate:
        return

    provider_kwargs = {}
    if args.use_vllm:
        provider_kwargs["use_vllm"] = True
    if args.provider == "vllm":
        provider_kwargs["base_url"] = args.vllm_url

    partial_eval_path = output_dir / "evaluation_results.json"
    evaluation = evaluate_queries(
        benchmark_image,
        generated["queries"],
        provider=args.provider,
        model_name=args.model,
        partial_output_path=partial_eval_path,
        **provider_kwargs,
    )
    eval_path = output_dir / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    print(f"Overall accuracy: {evaluation['overall']['accuracy']:.1%}")
    for task_type, stats in evaluation["by_task_type"].items():
        print(f"  {task_type}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['count']})")
    print(f"Evaluation saved to: {eval_path}")


if __name__ == "__main__":
    main()
