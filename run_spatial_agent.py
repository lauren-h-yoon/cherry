#!/usr/bin/env python3
"""
run_spatial_agent.py - Run the spatial reasoning agent evaluation

Examples:
    # Basic run (agent at z=0, target at max z)
    python run_spatial_agent.py --graph spatial_outputs/4r9OKorlcTk_spatial_graph.json

    # Custom start/target positions
    python run_spatial_agent.py --graph spatial_outputs/4r9OKorlcTk_spatial_graph.json --agent-z 2 --target-z 7

    # With multimodal (image) input
    python run_spatial_agent.py --graph spatial_outputs/4r9OKorlcTk_spatial_graph.json --multimodal

    # Generate annotated image only (no agent run)
    python run_spatial_agent.py --graph spatial_outputs/4r9OKorlcTk_spatial_graph.json --annotate-only
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from spatial_agent import SpatialAnnotator, SpatialReasoningAgent, AgentState


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
    """Run the spatial reasoning agent."""
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
    with open(results_path, 'w') as f:
        # Make serializable
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

    parser.add_argument(
        "--graph", "-g",
        required=True,
        help="Path to spatial_graph.json from Cherry pipeline"
    )
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

    if args.annotate_only:
        annotate_only(args)
    else:
        run_agent(args)


if __name__ == "__main__":
    main()
