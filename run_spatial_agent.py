#!/usr/bin/env python3
"""
run_spatial_agent.py - Run the spatial reasoning agent evaluation

Examples:
    # Basic navigation (agent at z=0, target at max z)
    python run_spatial_agent.py --graph spatial_outputs/4r9OKorlcTk_spatial_graph.json

    # Custom start/target positions
    python run_spatial_agent.py --graph spatial_outputs/4r9OKorlcTk_spatial_graph.json --agent-z 2 --target-z 7

    # With multimodal (image) input
    python run_spatial_agent.py --graph spatial_outputs/4r9OKorlcTk_spatial_graph.json --multimodal

    # Allocentric Q&A mode
    python run_spatial_agent.py --graph spatial_outputs/scene_spatial_graph.json \\
        --eval-mode qa --qa-question "Is the lamp to the left of the table?"

    # Navigation with allocentric goal
    python run_spatial_agent.py --graph spatial_outputs/scene_spatial_graph.json \\
        --eval-mode navigation --allocentric --context "Navigate to the object behind the table"

    # Perspective-shift mode
    python run_spatial_agent.py --graph spatial_outputs/scene_spatial_graph.json \\
        --eval-mode perspective --perspective-entity "table_0"

    # Full benchmark
    python run_spatial_agent.py --graph spatial_outputs/scene_spatial_graph.json \\
        --eval-mode benchmark --benchmark-tasks 30

    # Generate annotated image only (no agent run)
    python run_spatial_agent.py --graph spatial_outputs/4r9OKorlcTk_spatial_graph.json --annotate-only
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from spatial_agent import SpatialAnnotator, AgentState, get_agent
from spatial_agent.agent import AllocentricReasoningAgent

# Get agent class from lazy loader
SpatialReasoningAgent = get_agent()
from spatial_agent.allocentric import AllocentricRelationships
from spatial_agent.allocentric_eval import AllocentricEvaluator, AllocentricBenchmark


def annotate_only(args):
    """Generate annotated image without running agent."""
    annotator = SpatialAnnotator(args.graph, args.image)

    # Determine target z
    if args.target_z is None:
        args.target_z = max(wp.z_order for wp in annotator.waypoints)

    # Generate scenario
    output_path = Path(args.output_dir) / "annotated_scene.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img, config = annotator.generate_scenario(
        agent_z=args.agent_z,
        target_z=args.target_z,
        output_path=str(output_path)
    )

    print(f"\nAnnotated image saved to: {output_path}")
    print(f"\nScenario:")
    print(f"  Agent: {config.agent_position.name} (z={config.agent_position.z_order})")
    print(f"  Target: {config.target_position.name} (z={config.target_position.z_order})")
    print(f"\n  Waypoints ({len(config.waypoints)}):")
    for wp in sorted(config.waypoints, key=lambda w: w.z_order):
        print(f"    z={wp.z_order}: {wp.name} ({wp.category})")


def run_agent(args):
    """Run the spatial reasoning agent (egocentric navigation)."""
    # Check for API key
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    # Create agent
    agent = SpatialReasoningAgent(verbose=not args.quiet)

    # Setup scenario
    config = agent.setup_scenario(
        spatial_graph_path=args.graph,
        agent_z=args.agent_z,
        target_z=args.target_z,
        image_path=args.image,
        output_dir=args.output_dir
    )

    print(f"\n{'='*60}")
    print("SPATIAL REASONING EVALUATION")
    print(f"{'='*60}")
    print(f"Agent Start: {config.agent_position.name} (z={config.agent_position.z_order})")
    print(f"Target: {config.target_position.name} (z={config.target_position.z_order})")
    print(f"Waypoints: {len(config.waypoints)}")
    print(f"Annotated image: {agent.annotated_image_path}")
    print(f"{'='*60}\n")

    # Run agent
    if args.multimodal:
        print("Running with multimodal (image) input...\n")
        result = agent.run_with_image(
            additional_context=args.context or ""
        )
    else:
        print("Running with text-only input...\n")
        result = agent.run(
            include_image=True,
            additional_context=args.context or ""
        )

    # Print results
    print_navigation_results(result, agent, config, args)


def run_allocentric_navigation(args):
    """Run allocentric navigation with goal description."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Create allocentric agent
    agent = AllocentricReasoningAgent(verbose=not args.quiet)

    # Setup scenario
    config = agent.setup_scenario(
        spatial_graph_path=args.graph,
        agent_z=args.agent_z,
        target_z=args.target_z,
        image_path=args.image,
        output_dir=args.output_dir,
        evaluation_mode="navigation"
    )

    print(f"\n{'='*60}")
    print("ALLOCENTRIC NAVIGATION EVALUATION")
    print(f"{'='*60}")
    print(f"Mode: Navigation with allocentric goal")
    print(f"Agent Start: {config.agent_position.name} (z={config.agent_position.z_order})")
    if args.context:
        print(f"Goal: {args.context}")
    print(f"{'='*60}\n")

    # Run with allocentric context
    result = agent.run_with_image(
        additional_context=args.context or "",
        visual_feedback=True
    )

    # Print results
    print_navigation_results(result, agent, config, args)


def run_qa_evaluation(args):
    """Run Q&A evaluation mode."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    if not args.qa_question:
        print("Error: --qa-question is required for Q&A mode")
        sys.exit(1)

    # Create allocentric agent
    agent = AllocentricReasoningAgent(verbose=not args.quiet)

    # Setup scenario
    config = agent.setup_scenario(
        spatial_graph_path=args.graph,
        agent_z=args.agent_z,
        target_z=args.target_z,
        image_path=args.image,
        output_dir=args.output_dir,
        evaluation_mode="qa"
    )

    print(f"\n{'='*60}")
    print("ALLOCENTRIC Q&A EVALUATION")
    print(f"{'='*60}")
    print(f"Question: {args.qa_question}")
    print(f"{'='*60}\n")

    # Run Q&A task
    result = agent.run_qa_task(
        question=args.qa_question,
        include_image=True
    )

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Question: {result['question']}")
    print(f"Response: {result['response']}")
    print(f"Turns: {result['turns']}")

    # Save results
    results_path = Path(args.output_dir) / "qa_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {results_path}")


def run_perspective_evaluation(args):
    """Run perspective-shift evaluation mode."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    if not args.perspective_entity:
        print("Error: --perspective-entity is required for perspective mode")
        sys.exit(1)

    # Create allocentric agent
    agent = AllocentricReasoningAgent(verbose=not args.quiet)

    # Setup scenario
    config = agent.setup_scenario(
        spatial_graph_path=args.graph,
        agent_z=args.agent_z,
        target_z=args.target_z,
        image_path=args.image,
        output_dir=args.output_dir,
        evaluation_mode="perspective_shift"
    )

    print(f"\n{'='*60}")
    print("PERSPECTIVE-SHIFT EVALUATION")
    print(f"{'='*60}")
    print(f"Reference Entity: {args.perspective_entity}")
    print(f"{'='*60}\n")

    # Run perspective-shift task
    result = agent.run_perspective_shift_task(
        reference_entity=args.perspective_entity,
        direction=args.perspective_direction,
        include_image=True
    )

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Reference: {result['reference_entity']}")
    if result.get('direction'):
        print(f"Direction: {result['direction']}")
    print(f"Question: {result['question']}")
    print(f"\nResponse:\n{result['response']}")

    # Save results
    results_path = Path(args.output_dir) / "perspective_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {results_path}")


def run_benchmark(args):
    """Run full allocentric benchmark."""
    print(f"\n{'='*60}")
    print("ALLOCENTRIC REASONING BENCHMARK")
    print(f"{'='*60}")
    print(f"Graph: {args.graph}")
    print(f"Tasks per mode: {args.benchmark_tasks}")
    print(f"{'='*60}\n")

    # Create benchmark
    benchmark = AllocentricBenchmark(
        spatial_graph_path=args.graph,
        output_dir=args.output_dir
    )

    # Generate tasks
    print("Generating evaluation tasks...")
    tasks = benchmark.generate_all_tasks(
        qa_count=args.benchmark_tasks,
        nav_count=args.benchmark_tasks // 2,
        persp_count=args.benchmark_tasks // 2,
        seed=args.seed
    )

    # Print task summary
    print(f"\nGenerated tasks:")
    for mode, task_list in tasks.items():
        print(f"  {mode}: {len(task_list)} tasks")

    # Save tasks
    tasks_path = benchmark.save_tasks(tasks)

    # Show sample tasks
    print("\n" + "="*60)
    print("SAMPLE TASKS")
    print("="*60)

    for mode, task_list in tasks.items():
        print(f"\n{mode.upper()}:")
        for task in task_list[:2]:  # Show first 2
            print(f"  Q: {task.question}")
            print(f"  A: {task.ground_truth}")
            print(f"  Difficulty: {task.difficulty}")
            print()

    print("="*60)
    print(f"Tasks saved to: {tasks_path}")
    print("\nTo run evaluation with an agent, use the individual modes:")
    print(f"  --eval-mode qa --qa-question \"<question>\"")
    print(f"  --eval-mode perspective --perspective-entity \"<entity>\"")


def print_navigation_results(result, agent, config, args):
    """Print navigation results."""
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Reached Target: {'YES' if result['reached_target'] else 'NO'}")
    print(f"Total Moves: {result['total_moves']}")
    print(f"\nPath Taken:")

    path = result['path_summary']['path']
    for i, step in enumerate(path):
        arrow = "→" if i < len(path) - 1 else ""
        marker = " [START]" if i == 0 else (" [TARGET]" if step['id'] == agent.agent_state.target_waypoint_id else "")
        print(f"  {step['name']} (z={step['z_order']}){marker} {arrow}")

    # Save results
    results_path = Path(args.output_dir) / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        serializable_result = {
            'reached_target': result['reached_target'],
            'total_moves': result['total_moves'],
            'path_summary': result['path_summary'],
            'scenario': {
                'agent_start': config.agent_position.name,
                'agent_z': config.agent_position.z_order,
                'target': config.target_position.name,
                'target_z': config.target_position.z_order,
                'num_waypoints': len(config.waypoints)
            }
        }
        json.dump(serializable_result, f, indent=2)

    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run spatial reasoning agent evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--graph", "-g",
        required=True,
        help="Path to spatial_graph.json from Cherry pipeline"
    )

    # Basic configuration
    parser.add_argument(
        "--image", "-i",
        help="Override image path (default: from graph JSON)"
    )
    parser.add_argument(
        "--agent-z",
        type=int,
        default=0,
        help="Agent starting z-order (default: 0, closest)"
    )
    parser.add_argument(
        "--target-z",
        type=int,
        default=None,
        help="Target z-order (default: max, farthest)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="spatial_agent_outputs",
        help="Output directory for results"
    )

    # Evaluation mode
    parser.add_argument(
        "--eval-mode",
        choices=["navigation", "qa", "perspective", "benchmark"],
        default="navigation",
        help="Evaluation mode (default: navigation)"
    )

    # Allocentric options
    parser.add_argument(
        "--allocentric",
        action="store_true",
        help="Enable allocentric reasoning tools"
    )
    parser.add_argument(
        "--qa-question",
        help="Question for Q&A mode (e.g., 'Is the lamp behind the table?')"
    )
    parser.add_argument(
        "--perspective-entity",
        help="Entity for perspective-shift mode (e.g., 'table_0')"
    )
    parser.add_argument(
        "--perspective-direction",
        choices=["left", "right", "front", "behind", "above", "below"],
        help="Specific direction for perspective-shift"
    )
    parser.add_argument(
        "--benchmark-tasks",
        type=int,
        default=20,
        help="Number of tasks per mode for benchmark (default: 20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible task generation"
    )

    # Legacy/other options
    parser.add_argument(
        "--multimodal", "-m",
        action="store_true",
        help="Use multimodal (image) input"
    )
    parser.add_argument(
        "--annotate-only",
        action="store_true",
        help="Only generate annotated image, don't run agent"
    )
    parser.add_argument(
        "--context", "-c",
        help="Additional context/instructions for the agent"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Route to appropriate handler
    if args.annotate_only:
        annotate_only(args)
    elif args.eval_mode == "qa":
        run_qa_evaluation(args)
    elif args.eval_mode == "perspective":
        run_perspective_evaluation(args)
    elif args.eval_mode == "benchmark":
        run_benchmark(args)
    elif args.allocentric:
        run_allocentric_navigation(args)
    else:
        run_agent(args)


if __name__ == "__main__":
    main()
