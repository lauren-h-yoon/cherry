#!/usr/bin/env python3
"""
run_embodied_eval.py - Run embodied spatial intelligence evaluation

Evaluates VLM spatial intelligence using primitive tools as output mechanisms.
VLM sees image → reasons → produces outputs via tools → evaluated against ground truth.

Supports multiple VLM backends:
- Claude (Anthropic API) - default
- Qwen VL (local via transformers/vLLM)
- OpenAI GPT-4o
- Ollama (local)

Examples:
    # Run with Claude (default)
    python run_embodied_eval.py --graph spatial_outputs/scene_spatial_graph.json \
        --task suite --num-tasks 20

    # Run with Qwen VL (local)
    python run_embodied_eval.py --graph spatial_outputs/scene_spatial_graph.json \
        --task suite --provider qwen --model Qwen/Qwen2.5-VL-7B-Instruct

    # Run with Qwen VL using vLLM (faster)
    python run_embodied_eval.py --graph spatial_outputs/scene_spatial_graph.json \
        --task suite --provider qwen --use-vllm

    # Run with OpenAI GPT-4o
    python run_embodied_eval.py --graph spatial_outputs/scene_spatial_graph.json \
        --task suite --provider openai --model gpt-4o

    # Run with Ollama (local llava)
    python run_embodied_eval.py --graph spatial_outputs/scene_spatial_graph.json \
        --task suite --provider ollama --model llava

    # Run single localization task
    python run_embodied_eval.py --graph spatial_outputs/scene_spatial_graph.json \
        --task localization --object "table"

    # Run path planning task
    python run_embodied_eval.py --graph spatial_outputs/scene_spatial_graph.json \
        --task path --start "person" --goal "stool"

    # Run allocentric navigation
    python run_embodied_eval.py --graph spatial_outputs/scene_spatial_graph.json \
        --task allocentric --reference "table" --goal "lamp"
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from spatial_agent.ground_truth import GroundTruth
from spatial_agent.tasks import TaskGenerator, SpatialTask, TaskType, Difficulty
from spatial_agent.embodied_agent import EmbodiedSpatialAgent, EmbodiedEvaluationRunner
from spatial_agent.model_providers import list_available_providers


def get_provider_kwargs(args):
    """Extract provider-specific kwargs from args."""
    kwargs = {}
    if hasattr(args, 'use_vllm') and args.use_vllm:
        kwargs['use_vllm'] = True
    if hasattr(args, 'vllm_url') and args.provider == 'vllm':
        kwargs['base_url'] = args.vllm_url
    return kwargs


def run_localization(args):
    """Run object localization task."""
    gt = GroundTruth(args.graph)
    generator = TaskGenerator(gt, image_path=args.image, seed=args.seed)

    task = generator.generate_localization_task(args.object)

    agent = EmbodiedSpatialAgent(
        provider=args.provider,
        model_name=args.model,
        verbose=not args.quiet,
        **get_provider_kwargs(args)
    )
    report, _, _ = agent.run_and_evaluate(task, args.image, gt)

    print(f"\nResult: {'PASS' if report.overall_score > 0.5 else 'FAIL'}")
    print(f"Score: {report.overall_score:.1%}")

    save_results(args, {'task': task.to_dict(), 'report': report.to_dict()})


def run_path_planning(args):
    """Run allocentric path planning task (object-to-object)."""
    gt = GroundTruth(args.graph)
    generator = TaskGenerator(gt, image_path=args.image, seed=args.seed)

    task = generator.generate_allocentric_path_task(args.start, args.goal)

    agent = EmbodiedSpatialAgent(
        provider=args.provider,
        model_name=args.model,
        verbose=not args.quiet,
        **get_provider_kwargs(args)
    )
    report, _, _ = agent.run_and_evaluate(task, args.image, gt)

    print(f"\nResult: {'PASS' if report.overall_score > 0.5 else 'FAIL'}")
    print(f"Score: {report.overall_score:.1%}")

    save_results(args, {'task': task.to_dict(), 'report': report.to_dict()})


def run_egocentric(args):
    """Run egocentric path planning task (viewer-relative direction)."""
    gt = GroundTruth(args.graph)
    generator = TaskGenerator(gt, image_path=args.image, seed=args.seed)

    # Use direction if provided, otherwise use default
    direction = getattr(args, 'direction', None)

    task = generator.generate_egocentric_path_task(direction=direction)

    agent = EmbodiedSpatialAgent(
        provider=args.provider,
        model_name=args.model,
        verbose=not args.quiet,
        **get_provider_kwargs(args)
    )
    report, _, _ = agent.run_and_evaluate(task, args.image, gt)

    print(f"\nResult: {'PASS' if report.overall_score > 0.5 else 'FAIL'}")
    print(f"Score: {report.overall_score:.1%}")

    save_results(args, {'task': task.to_dict(), 'report': report.to_dict()})


def run_allocentric(args):
    """Run allocentric path planning task (from reference object to goal)."""
    gt = GroundTruth(args.graph)
    generator = TaskGenerator(gt, image_path=args.image, seed=args.seed)

    task = generator.generate_allocentric_path_task(args.reference, args.goal)

    agent = EmbodiedSpatialAgent(
        provider=args.provider,
        model_name=args.model,
        verbose=not args.quiet,
        **get_provider_kwargs(args)
    )
    report, _, _ = agent.run_and_evaluate(task, args.image, gt)

    print(f"\nResult: {'PASS' if report.overall_score > 0.5 else 'FAIL'}")
    print(f"Score: {report.overall_score:.1%}")

    save_results(args, {'task': task.to_dict(), 'report': report.to_dict()})


def run_relation(args):
    """Run allocentric spatial QA task."""
    gt = GroundTruth(args.graph)
    generator = TaskGenerator(gt, image_path=args.image, seed=args.seed)

    # Generate allocentric QA task
    task = generator.generate_allocentric_qa_task(
        relation=args.relation_type,
        entity_a=args.entity_a,
        entity_b=args.entity_b
    )

    agent = EmbodiedSpatialAgent(
        provider=args.provider,
        model_name=args.model,
        verbose=not args.quiet,
        **get_provider_kwargs(args)
    )
    report, _, _ = agent.run_and_evaluate(task, args.image, gt)

    print(f"\nResult: {'PASS' if report.overall_score > 0.5 else 'FAIL'}")
    print(f"Score: {report.overall_score:.1%}")

    save_results(args, {'task': task.to_dict(), 'report': report.to_dict()})


def run_egocentric_qa(args):
    """Run egocentric spatial QA task."""
    gt = GroundTruth(args.graph)
    generator = TaskGenerator(gt, image_path=args.image, seed=args.seed)

    direction = getattr(args, 'direction', None)
    task = generator.generate_egocentric_qa_task(direction=direction)

    agent = EmbodiedSpatialAgent(
        provider=args.provider,
        model_name=args.model,
        verbose=not args.quiet,
        **get_provider_kwargs(args)
    )
    report, _, _ = agent.run_and_evaluate(task, args.image, gt)

    print(f"\nResult: {'PASS' if report.overall_score > 0.5 else 'FAIL'}")
    print(f"Score: {report.overall_score:.1%}")

    save_results(args, {'task': task.to_dict(), 'report': report.to_dict()})


def run_suite(args):
    """Run full evaluation suite."""
    gt = GroundTruth(args.graph)
    generator = TaskGenerator(gt, image_path=args.image, seed=args.seed)

    # Generate mixed suite with 5 task types:
    # - Object Localization
    # - Egocentric Path Planning
    # - Allocentric Path Planning
    # - Egocentric Spatial QA
    # - Allocentric Spatial QA
    tasks = generator.generate_mixed_suite(total_tasks=args.num_tasks)

    print(f"\nRunning {len(tasks)} tasks...")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or 'default'}")
    print(f"Image: {args.image}")
    print(f"Graph: {args.graph}")

    runner = EmbodiedEvaluationRunner(
        args.graph,
        args.image,
        provider=args.provider,
        model_name=args.model,
        verbose=not args.quiet,
        **get_provider_kwargs(args)
    )

    results = runner.run_task_suite(tasks)
    runner.print_summary(results)

    save_results(args, results)


def save_results(args, results):
    """Save results to output file."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "embodied_eval_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def print_available_models():
    """Print available providers and models."""
    providers = list_available_providers()
    print("\nAvailable VLM Providers and Models:")
    print("="*50)
    for provider, models in providers.items():
        print(f"\n  {provider}:")
        for model in models:
            print(f"    - {model}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Embodied Spatial Intelligence Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--graph", "-g",
        required=True,
        help="Path to spatial_graph.json from SAM3+Depth pipeline"
    )

    parser.add_argument(
        "--image", "-i",
        help="Path to scene image (default: from graph JSON)"
    )

    # Model selection
    parser.add_argument(
        "--provider", "-p",
        choices=["claude", "qwen", "vllm", "openai", "ollama"],
        default="claude",
        help="VLM provider (default: claude). Use 'vllm' for vLLM server with tool calling."
    )
    parser.add_argument(
        "--model", "-m",
        help="Specific model name (uses provider default if not specified)"
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM server URL (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for faster local inference (Qwen transformers only)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available providers and models"
    )

    # Task type
    parser.add_argument(
        "--task", "-t",
        choices=["localization", "path", "egocentric", "allocentric",
                 "relation", "perspective", "suite"],
        default="suite",
        help="Task type to run (default: suite)"
    )

    # Task-specific arguments
    parser.add_argument("--object", help="Target object for localization")
    parser.add_argument("--start", help="Start object for path/navigation")
    parser.add_argument("--goal", help="Goal object for path/navigation")
    parser.add_argument("--reference", help="Reference object for allocentric/perspective")
    parser.add_argument("--start-x", type=int, help="Start X for egocentric")
    parser.add_argument("--start-y", type=int, help="Start Y for egocentric")
    parser.add_argument("--direction", help="Direction for perspective (left/right/front/behind)")
    parser.add_argument("--relation-type", help="Relation type for QA (left_of, behind, etc.)")
    parser.add_argument("--relation", help="Relation for pointing task")
    parser.add_argument("--entity-a", help="Entity A for relation QA")
    parser.add_argument("--entity-b", help="Entity B for relation QA")

    # Suite options
    parser.add_argument(
        "--num-tasks", "-n",
        type=int,
        default=20,
        help="Number of tasks for suite (default: 20)"
    )

    # General options
    parser.add_argument(
        "--output-dir", "-o",
        default="embodied_eval_outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print_available_models()
        sys.exit(0)

    # Get image path from graph if not specified
    if not args.image:
        with open(args.graph) as f:
            graph_data = json.load(f)
        args.image = graph_data.get('source_image', '')
        if not args.image:
            print("Error: No image path in graph and --image not specified")
            sys.exit(1)

    # Check files exist
    if not Path(args.graph).exists():
        print(f"Error: Graph file not found: {args.graph}")
        sys.exit(1)
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    # Check API keys based on provider
    import os
    if args.provider == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            print("Required for Claude provider")
            sys.exit(1)
    elif args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Required for OpenAI provider")
            sys.exit(1)
    elif args.provider == "vllm":
        print(f"Using vLLM server at: {args.vllm_url}")
        print("Make sure vLLM is running with:")
        print(f"  vllm serve {args.model or 'Qwen/Qwen3-8B'} --enable-auto-tool-choice --tool-call-parser hermes")
    elif args.provider == "qwen":
        print("Using Qwen VL (local inference via transformers)")
        if args.use_vllm:
            print("Using vLLM backend for faster inference")
    elif args.provider == "ollama":
        print("Using Ollama (local inference)")

    # Route to task handler
    if args.task == "localization":
        run_localization(args)
    elif args.task == "path":
        run_path_planning(args)
    elif args.task == "egocentric":
        run_egocentric(args)
    elif args.task == "allocentric":
        run_allocentric(args)
    elif args.task == "relation":
        run_relation(args)
    elif args.task == "perspective":
        run_perspective(args)
    elif args.task == "suite":
        run_suite(args)


if __name__ == "__main__":
    main()
