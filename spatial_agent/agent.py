#!/usr/bin/env python3
"""
agent.py - LangChain agent for spatial reasoning

Embodied agent that navigates through waypoints to reach a target,
reasoning about z-order and occlusion.
"""

import os
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from .state import AgentState
from .tools import create_tools
from .annotator import SpatialAnnotator, SceneConfig


SYSTEM_PROMPT = """You are an embodied spatial reasoning agent navigating through a 3D scene.

## Your Situation
- You are positioned at a specific location in a scene (marked as AGENT)
- You need to navigate to a target location (marked as TARGET/star)
- The scene contains waypoints at different depths (z-orders)
- z=0 is closest to the camera, higher z values are farther away

## Your Tools
1. **get_waypoints**: See all available waypoints with their z-order and positions
2. **move_to**: Move to a specific waypoint
3. **rotate**: Rotate your view to better understand spatial relationships
4. **scale**: Zoom in/out to see depth relationships more clearly

## Key Concepts
- **Z-order**: Depth ordering. Lower z = closer to camera, higher z = farther
- **Occlusion**: Objects in front can block access to objects behind. You cannot pass through objects.
- **Valid paths**: You should reason about which waypoints you can reach without passing through obstacles

## Your Task
1. First, use get_waypoints to understand the scene layout
2. Analyze the z-order relationships and identify potential paths
3. Consider using rotate/scale if you need to better understand depth relationships
4. Plan a path from your current position to the target
5. Execute the path using move_to commands
6. Explain your reasoning at each step

## Important
- Think carefully about occlusion - you cannot pass through objects
- Consider the z-order when planning your path
- A valid path respects the 3D structure of the scene
- Explain why you chose your path and why it avoids obstacles

Begin by examining the waypoints and planning your approach."""


class SpatialReasoningAgent:
    """
    LangChain-based agent for spatial reasoning tasks.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        verbose: bool = True
    ):
        """
        Initialize the spatial reasoning agent.

        Args:
            model_name: Anthropic model to use
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            temperature: Model temperature
            verbose: Whether to print agent steps
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.llm = ChatAnthropic(
            model=model_name,
            anthropic_api_key=self.api_key,
            temperature=temperature,
            max_tokens=4096
        )

        self.verbose = verbose
        self.agent_state: Optional[AgentState] = None
        self.agent_graph = None
        self.annotated_image_path: Optional[str] = None

    def setup_scenario(
        self,
        spatial_graph_path: str,
        agent_z: int = 0,
        target_z: Optional[int] = None,
        image_path: Optional[str] = None,
        output_dir: str = "spatial_agent_outputs"
    ) -> SceneConfig:
        """
        Set up a spatial reasoning scenario.

        Args:
            spatial_graph_path: Path to spatial_graph.json
            agent_z: Starting z-order for agent
            target_z: Target z-order (default: farthest)
            image_path: Optional override for image path
            output_dir: Directory for output files

        Returns:
            SceneConfig for the scenario
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create annotator and scene config
        annotator = SpatialAnnotator(spatial_graph_path, image_path)

        if target_z is None:
            target_z = max(wp.z_order for wp in annotator.waypoints)

        scene_config = annotator.create_scene_config(
            agent_z=agent_z,
            target_z=target_z
        )

        # Generate annotated image
        self.annotated_image_path = str(output_dir / "scenario_annotated.png")
        annotator.annotate(scene_config, self.annotated_image_path)

        # Create agent state
        self.agent_state = AgentState.from_scene_config(scene_config)

        # Create tools
        tools = create_tools(self.agent_state)

        # Create agent using langgraph
        self.agent_graph = create_react_agent(
            self.llm,
            tools,
            prompt=SYSTEM_PROMPT
        )

        return scene_config

    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64 for vision input."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def run(
        self,
        include_image: bool = True,
        additional_context: str = ""
    ) -> Dict[str, Any]:
        """
        Run the spatial reasoning agent.

        Args:
            include_image: Whether to include the annotated image
            additional_context: Additional instructions or context

        Returns:
            Dict with results including path taken and reasoning
        """
        if not self.agent_graph or not self.agent_state:
            raise ValueError("Must call setup_scenario first")

        # Build input message
        input_text = f"""You are in a scene and need to navigate to the target.

Current position: {self.agent_state.get_current_waypoint_info().get('name')} (z={self.agent_state.current_z_order})
Target: {self.agent_state.get_target_waypoint_info().get('name')} (z={self.agent_state.target_z_order})

{additional_context}

Please analyze the scene, plan a path, and navigate to the target. Explain your reasoning about z-order and occlusion as you go."""

        if include_image and self.annotated_image_path:
            input_text += f"\n\n[Annotated scene image available at: {self.annotated_image_path}]"
            input_text += "\nThe image shows waypoints as colored dots (green=near, red=far), your position as a green triangle, and the target as a yellow star."

        # Run agent
        messages = [HumanMessage(content=input_text)]

        result = self.agent_graph.invoke(
            {"messages": messages},
            {"recursion_limit": 25}
        )

        # Get final output
        final_messages = result.get("messages", [])
        output = ""
        for msg in final_messages:
            if isinstance(msg, AIMessage):
                output = msg.content

        # Get path summary
        path_summary = self.agent_state.get_path_summary()

        return {
            "output": output,
            "messages": final_messages,
            "path_summary": path_summary,
            "reached_target": self.agent_state.reached_target,
            "total_moves": self.agent_state.move_count,
            "final_state": self.agent_state.to_dict()
        }

    def run_with_image(
        self,
        additional_context: str = ""
    ) -> Dict[str, Any]:
        """
        Run with multimodal input (image + text).

        This uses direct Anthropic API for multimodal support.
        """
        if not self.agent_state:
            raise ValueError("Must call setup_scenario first")

        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)

        # Encode image
        image_data = self._encode_image(self.annotated_image_path)

        # Build tool descriptions
        tools = create_tools(self.agent_state)
        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}" for t in tools
        ])

        messages = []

        # Initial message with image
        initial_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data
                }
            },
            {
                "type": "text",
                "text": f"""Look at this annotated scene image. You are the AGENT (green triangle) and need to reach the TARGET (yellow star).

The colored dots are waypoints:
- Green dots = closer to camera (lower z-order)
- Red dots = farther from camera (higher z-order)
- Each waypoint is labeled with its z-order (e.g., z=0, z=1, etc.)

Your current position: {self.agent_state.get_current_waypoint_info().get('name')} (z={self.agent_state.current_z_order})
Your target: {self.agent_state.get_target_waypoint_info().get('name')} (z={self.agent_state.target_z_order})

{additional_context}

Available tools:
{tool_descriptions}

To use a tool, respond with:
TOOL: <tool_name>
ARGS: <arguments as JSON>

For example:
TOOL: get_waypoints
ARGS: {{"include_current": true}}

Or:
TOOL: move_to
ARGS: {{"waypoint_id": "entity_3"}}

Analyze the scene, understand the spatial relationships, plan a path avoiding occlusions, and navigate to the target.
Explain your reasoning about depth and occlusion at each step."""
            }
        ]

        messages.append({"role": "user", "content": initial_content})

        # Conversation loop
        max_turns = 15
        conversation_log = []

        for turn in range(max_turns):
            # Get response
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=messages
            )

            assistant_text = response.content[0].text
            conversation_log.append({"role": "assistant", "content": assistant_text})

            if self.verbose:
                print(f"\n=== Turn {turn + 1} ===")
                print(assistant_text)

            # Check for tool use
            if "TOOL:" in assistant_text:
                # Parse tool call
                lines = assistant_text.split("\n")
                tool_name = None
                tool_args = {}

                for line in lines:
                    if line.startswith("TOOL:"):
                        tool_name = line.replace("TOOL:", "").strip()
                    elif line.startswith("ARGS:"):
                        import json
                        args_str = line.replace("ARGS:", "").strip()
                        try:
                            tool_args = json.loads(args_str)
                        except:
                            tool_args = {}

                # Execute tool
                if tool_name:
                    tool_result = self._execute_tool(tool_name, tool_args)

                    if self.verbose:
                        print(f"\n[Tool Result: {tool_name}]")
                        print(tool_result[:500] + "..." if len(tool_result) > 500 else tool_result)

                    # Add to conversation
                    messages.append({"role": "assistant", "content": assistant_text})
                    messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})
                    conversation_log.append({"role": "tool", "name": tool_name, "result": tool_result})

                    # Check if reached target
                    if self.agent_state.reached_target:
                        break
            else:
                # No tool call, add response and prompt for action
                messages.append({"role": "assistant", "content": assistant_text})

                if self.agent_state.reached_target:
                    break

                messages.append({
                    "role": "user",
                    "content": "Please use a tool to continue navigating, or explain if you've completed the task."
                })

        # Get path summary
        path_summary = self.agent_state.get_path_summary()

        return {
            "conversation": conversation_log,
            "path_summary": path_summary,
            "reached_target": self.agent_state.reached_target,
            "total_moves": self.agent_state.move_count,
            "final_state": self.agent_state.to_dict()
        }

    def _execute_tool(self, tool_name: str, args: Dict) -> str:
        """Execute a tool by name."""
        if tool_name == "get_waypoints":
            include_current = args.get("include_current", True)
            waypoints = self.agent_state.get_all_waypoints()

            result = "Available waypoints (sorted by z-order, front to back):\n\n"
            for wp in waypoints:
                markers = []
                if wp['id'] == self.agent_state.current_waypoint_id:
                    if not include_current:
                        continue
                    markers.append("CURRENT")
                if wp['id'] == self.agent_state.target_waypoint_id:
                    markers.append("TARGET")

                marker_str = f" [{', '.join(markers)}]" if markers else ""
                result += f"- {wp['id']}: {wp['name']}{marker_str}\n"
                result += f"    z_order: {wp['z_order']}, depth: {wp['relative_depth']:.3f}\n"
                result += f"    position: ({wp['position'][0]:.0f}, {wp['position'][1]:.0f})\n\n"

            return result

        elif tool_name == "move_to":
            waypoint_id = args.get("waypoint_id", "")
            success, message = self.agent_state.move_to(waypoint_id)
            return f"{'SUCCESS' if success else 'FAILED'}: {message}\n\n{self.agent_state.get_status()}"

        elif tool_name == "rotate":
            angle = args.get("angle", 0)
            return self.agent_state.rotate_view(angle)

        elif tool_name == "scale":
            factor = args.get("factor", 1.0)
            return self.agent_state.scale_view(factor)

        else:
            return f"Unknown tool: {tool_name}"


def main():
    """Demo: Run spatial reasoning agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Run spatial reasoning agent")
    parser.add_argument("--graph", "-g", required=True, help="Path to spatial_graph.json")
    parser.add_argument("--image", "-i", help="Override image path")
    parser.add_argument("--agent-z", type=int, default=0, help="Agent starting z-order")
    parser.add_argument("--target-z", type=int, help="Target z-order")
    parser.add_argument("--output-dir", "-o", default="spatial_agent_outputs")
    parser.add_argument("--multimodal", action="store_true", help="Use multimodal (image) input")

    args = parser.parse_args()

    # Create agent
    agent = SpatialReasoningAgent(verbose=True)

    # Setup scenario
    config = agent.setup_scenario(
        spatial_graph_path=args.graph,
        agent_z=args.agent_z,
        target_z=args.target_z,
        image_path=args.image,
        output_dir=args.output_dir
    )

    print(f"\n{'='*60}")
    print("Scenario Setup")
    print(f"{'='*60}")
    print(f"Agent: {config.agent_position.name} (z={config.agent_position.z_order})")
    print(f"Target: {config.target_position.name} (z={config.target_position.z_order})")
    print(f"Waypoints: {len(config.waypoints)}")
    print(f"Annotated image: {agent.annotated_image_path}")
    print(f"{'='*60}\n")

    # Run agent
    if args.multimodal:
        result = agent.run_with_image()
    else:
        result = agent.run()

    # Print results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Reached target: {result['reached_target']}")
    print(f"Total moves: {result['total_moves']}")
    print(f"Path: {result['path_summary']}")


if __name__ == "__main__":
    main()
