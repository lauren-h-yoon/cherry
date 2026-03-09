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
from .allocentric import AllocentricRelationships
from .allocentric_tools import create_allocentric_tools


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

## Visual Feedback
After each move, you will receive an UPDATED IMAGE showing:
- Your NEW position (green AGENT triangle)
- Visited waypoints (purple circles with checkmarks ✓)
- The path you've taken (purple arrows connecting visited points)
- Remaining unvisited waypoints (colored by depth)

Use this visual feedback to:
- Verify you moved to the correct location
- Track your progress toward the target
- Adjust your strategy based on what you observe

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
6. After each move, OBSERVE the visual feedback to verify your progress
7. Explain your reasoning at each step

## Important
- Think carefully about occlusion - you cannot pass through objects
- Consider the z-order when planning your path
- A valid path respects the 3D structure of the scene
- Explain why you chose your path and why it avoids obstacles
- USE the visual feedback images to confirm your moves and adjust your plan

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
        self.annotator: Optional[SpatialAnnotator] = None
        self.output_dir: Optional[Path] = None
        self.target_waypoint_id: Optional[str] = None

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
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create annotator and scene config
        self.annotator = SpatialAnnotator(spatial_graph_path, image_path)

        if target_z is None:
            target_z = max(wp.z_order for wp in self.annotator.waypoints)

        scene_config = self.annotator.create_scene_config(
            agent_z=agent_z,
            target_z=target_z
        )

        # Store target waypoint ID for rendering
        self.target_waypoint_id = scene_config.target_position.id

        # Generate annotated image
        self.annotated_image_path = str(self.output_dir / "scenario_annotated.png")
        self.annotator.annotate(scene_config, self.annotated_image_path)

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

    def _render_current_state(self, move_number: Optional[int] = None) -> str:
        """
        Render the current agent state and return base64 encoded image.

        Args:
            move_number: Optional move number for labeling

        Returns:
            Base64 encoded PNG image
        """
        if not self.annotator or not self.agent_state:
            raise ValueError("Must call setup_scenario first")

        # Render current state
        output_path = str(self.output_dir / f"state_move_{move_number or 0}.png")
        img = self.annotator.render_state(
            self.agent_state,
            self.target_waypoint_id,
            output_path=output_path,
            move_number=move_number
        )

        # Encode to base64
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.standard_b64encode(buffer.read()).decode("utf-8")

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
        additional_context: str = "",
        visual_feedback: bool = True
    ) -> Dict[str, Any]:
        """
        Run with multimodal input (image + text).

        This uses direct Anthropic API for multimodal support.
        With visual_feedback=True, the agent receives updated images after each move.

        Args:
            additional_context: Additional instructions for the agent
            visual_feedback: If True, send updated rendered image after each move

        Returns:
            Dict with conversation log, path summary, and results
        """
        if not self.agent_state:
            raise ValueError("Must call setup_scenario first")

        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)

        # Encode initial image
        image_data = self._encode_image(self.annotated_image_path)

        # Build tool descriptions
        tools = create_tools(self.agent_state)
        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}" for t in tools
        ])

        messages = []

        # Visual feedback instructions
        visual_feedback_note = ""
        if visual_feedback:
            visual_feedback_note = """
IMPORTANT: After each move, you will receive an updated image showing:
- Your NEW position (green triangle)
- Visited waypoints (purple with checkmarks)
- The path you've taken (purple arrows)
- Remaining unvisited waypoints
Use this visual feedback to verify your progress and adjust your strategy."""

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
{visual_feedback_note}

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

                    # Add assistant message
                    messages.append({"role": "assistant", "content": assistant_text})

                    # Build response with optional visual feedback
                    if visual_feedback and tool_name == "move_to":
                        # Render updated state image
                        updated_image_data = self._render_current_state(
                            move_number=self.agent_state.move_count
                        )

                        if self.verbose:
                            print(f"[Visual Feedback: Rendered state after move #{self.agent_state.move_count}]")

                        # Send image + text response
                        feedback_content = [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": updated_image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": f"""Tool result:
{tool_result}

[VISUAL FEEDBACK] Here is the updated scene showing your current position after the move.
- Your position is marked with the green AGENT triangle
- Purple waypoints with checkmarks (✓) are places you've already visited
- Purple arrows show the path you've taken so far

Analyze this visual feedback and continue navigating to the TARGET."""
                            }
                        ]
                        messages.append({"role": "user", "content": feedback_content})
                        conversation_log.append({
                            "role": "tool",
                            "name": tool_name,
                            "result": tool_result,
                            "visual_feedback": True,
                            "move_number": self.agent_state.move_count
                        })
                    else:
                        # Text-only response for non-move tools
                        messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})
                        conversation_log.append({"role": "tool", "name": tool_name, "result": tool_result})

                    # Check if reached target
                    if self.agent_state.reached_target:
                        # Render final state
                        if visual_feedback:
                            self._render_current_state(move_number=self.agent_state.move_count)
                            if self.verbose:
                                print(f"[Visual Feedback: Rendered final state - TARGET REACHED!]")
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


# ========== Allocentric Reasoning Agent ==========

ALLOCENTRIC_SYSTEM_PROMPT = """

## Allocentric Reasoning Capabilities

In addition to z-order navigation, you can reason about object-to-object spatial relationships:

### Spatial Relations
- **Left/Right**: Horizontal position in the scene (from camera's view)
- **Front/Behind**: Depth relationship (front = closer to camera, behind = farther)
- **Above/Below**: Vertical position in the scene
- **Between**: Whether an object is spatially between two others
- **Near/Far**: Proximity between objects

### Additional Tools
- **get_spatial_relation**: Query pairwise relationships between any two entities
- **query_scene**: Answer natural language spatial questions
- **perspective_shift**: Describe the scene from any object's viewpoint
- **resolve_allocentric_goal**: Convert allocentric goals (e.g., "object behind the table") to waypoint IDs

### Evaluation Modes
You may be asked to:
1. **Q&A**: Answer spatial questions directly (e.g., "Is the lamp behind the table?")
2. **Navigation**: Navigate to allocentrically-described targets (e.g., "Go to the object behind the table")
3. **Perspective-Shift**: Describe what you would see from another object's position

### Important Notes
- Always reason about spatial relationships explicitly
- Use the allocentric tools to verify your understanding
- Explain your spatial reasoning step by step
- For perspective shifts, imagine standing at the reference object and looking around
"""


class AllocentricReasoningAgent(SpatialReasoningAgent):
    """
    Extended agent with allocentric spatial reasoning capabilities.

    Supports:
    - Q&A evaluation: Answer spatial relationship questions
    - Navigation with allocentric goals: Navigate to "object behind table"
    - Perspective-shift tasks: Describe view from object's position
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allocentric: Optional[AllocentricRelationships] = None
        self.evaluation_mode: str = "navigation"

    def setup_scenario(
        self,
        spatial_graph_path: str,
        agent_z: int = 0,
        target_z: Optional[int] = None,
        image_path: Optional[str] = None,
        output_dir: str = "spatial_agent_outputs",
        evaluation_mode: str = "navigation"
    ) -> SceneConfig:
        """
        Set up a spatial reasoning scenario with allocentric capabilities.

        Args:
            spatial_graph_path: Path to spatial_graph.json
            agent_z: Starting z-order for agent
            target_z: Target z-order (default: farthest)
            image_path: Optional override for image path
            output_dir: Directory for output files
            evaluation_mode: 'navigation', 'qa', or 'perspective_shift'

        Returns:
            SceneConfig for the scenario
        """
        # Call parent setup
        config = super().setup_scenario(
            spatial_graph_path=spatial_graph_path,
            agent_z=agent_z,
            target_z=target_z,
            image_path=image_path,
            output_dir=output_dir
        )

        # Initialize allocentric relationships
        self.allocentric = AllocentricRelationships(spatial_graph_path)
        self.evaluation_mode = evaluation_mode

        # Create combined tools
        base_tools = create_tools(self.agent_state)
        allocentric_tools = create_allocentric_tools(
            self.allocentric,
            self.agent_state
        )
        all_tools = base_tools + allocentric_tools

        # Combine system prompts
        combined_prompt = SYSTEM_PROMPT + ALLOCENTRIC_SYSTEM_PROMPT

        # Recreate agent with extended tools
        self.agent_graph = create_react_agent(
            self.llm,
            all_tools,
            prompt=combined_prompt
        )

        return config

    def run_qa_task(
        self,
        question: str,
        include_image: bool = True
    ) -> Dict[str, Any]:
        """
        Run a Q&A evaluation task.

        This is a single-turn Q&A without navigation - the agent
        simply answers a spatial relationship question.

        Args:
            question: The spatial question to answer
            include_image: Whether to include the annotated image

        Returns:
            Dict with response and evaluation info
        """
        if not self.agent_state:
            raise ValueError("Must call setup_scenario first")

        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)

        # Build message
        content = []

        if include_image and self.annotated_image_path:
            image_data = self._encode_image(self.annotated_image_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data
                }
            })

        content.append({
            "type": "text",
            "text": f"""Look at this scene and answer the following spatial reasoning question.

Question: {question}

Use the available tools if needed to verify spatial relationships.
Provide a clear answer (yes/no if applicable) and explain your reasoning about the spatial relationships.
"""
        })

        # Get scene info for context
        scene_summary = self.allocentric.get_scene_summary()
        tool_descriptions = self._get_tool_descriptions()

        qa_system_prompt = f"""You are a spatial reasoning assistant answering questions about a scene.

Scene Info:
- {scene_summary['entity_count']} entities
- Depth range: z=0 (closest to camera) to z={scene_summary['depth_range'][1]} (farthest)

{tool_descriptions}

Answer the question clearly and explain your spatial reasoning."""

        messages = [{"role": "user", "content": content}]

        # Single-turn response (may use tools)
        max_turns = 5
        for turn in range(max_turns):
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=qa_system_prompt,
                messages=messages
            )

            assistant_text = response.content[0].text

            if self.verbose:
                print(f"\n=== Turn {turn + 1} ===")
                print(assistant_text)

            # Check for tool use
            if "TOOL:" in assistant_text:
                tool_name, tool_args = self._parse_tool_call(assistant_text)
                if tool_name:
                    tool_result = self._execute_tool(tool_name, tool_args)

                    if self.verbose:
                        print(f"\n[Tool: {tool_name}]")
                        print(tool_result[:300] + "..." if len(tool_result) > 300 else tool_result)

                    messages.append({"role": "assistant", "content": assistant_text})
                    messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})
                    continue

            # No tool call - this is the final answer
            break

        return {
            "question": question,
            "response": assistant_text,
            "turns": turn + 1
        }

    def run_perspective_shift_task(
        self,
        reference_entity: str,
        direction: Optional[str] = None,
        include_image: bool = True
    ) -> Dict[str, Any]:
        """
        Run a perspective-shift evaluation task.

        The agent describes what would be visible from a specific
        object's viewpoint.

        Args:
            reference_entity: Entity to view the scene from
            direction: Optional specific direction to describe
            include_image: Whether to include the annotated image

        Returns:
            Dict with response and evaluation info
        """
        if not self.agent_state:
            raise ValueError("Must call setup_scenario first")

        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)

        # Resolve entity
        entity_id = self.allocentric.resolve_entity(reference_entity)
        if not entity_id:
            return {"error": f"Could not find entity: {reference_entity}"}

        entity_name = self.allocentric.nodes[entity_id].get('name', entity_id)

        # Build question
        if direction:
            question = f"Imagine you are standing at the {entity_name}. What would be to your {direction}?"
        else:
            question = f"Imagine you are standing at the {entity_name}. Describe what you would see in all directions (left, right, in front, behind)."

        # Build message
        content = []

        if include_image and self.annotated_image_path:
            image_data = self._encode_image(self.annotated_image_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data
                }
            })

        content.append({
            "type": "text",
            "text": f"""This is a perspective-shift task. You need to mentally position yourself at a specific object and describe the scene from that viewpoint.

{question}

Use the perspective_shift tool to help verify your answer.
Remember: "in front" means closer to the camera, "behind" means farther from the camera.
"""
        })

        tool_descriptions = self._get_tool_descriptions()

        perspective_system_prompt = f"""You are a spatial reasoning assistant performing a perspective-shift task.

You need to imagine being positioned at a specific object in the scene and describe what you would see from that viewpoint.

{tool_descriptions}

Describe the scene from the requested perspective, listing objects in each direction."""

        messages = [{"role": "user", "content": content}]

        # Allow multiple turns for tool use
        max_turns = 5
        final_response = ""

        for turn in range(max_turns):
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=perspective_system_prompt,
                messages=messages
            )

            assistant_text = response.content[0].text

            if self.verbose:
                print(f"\n=== Turn {turn + 1} ===")
                print(assistant_text)

            # Check for tool use
            if "TOOL:" in assistant_text:
                tool_name, tool_args = self._parse_tool_call(assistant_text)
                if tool_name:
                    tool_result = self._execute_tool(tool_name, tool_args)

                    if self.verbose:
                        print(f"\n[Tool: {tool_name}]")
                        print(tool_result[:300] + "..." if len(tool_result) > 300 else tool_result)

                    messages.append({"role": "assistant", "content": assistant_text})
                    messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})
                    continue

            final_response = assistant_text
            break

        return {
            "reference_entity": reference_entity,
            "direction": direction,
            "question": question,
            "response": final_response,
            "turns": turn + 1
        }

    def run_navigation_with_allocentric_goal(
        self,
        goal_description: str,
        visual_feedback: bool = True
    ) -> Dict[str, Any]:
        """
        Run navigation with an allocentric goal description.

        The agent must first resolve the allocentric goal to a waypoint,
        then navigate to it.

        Args:
            goal_description: e.g., "the object behind the table"
            visual_feedback: Whether to provide visual feedback after moves

        Returns:
            Dict with path summary and evaluation info
        """
        if not self.agent_state:
            raise ValueError("Must call setup_scenario first")

        # Build navigation prompt with allocentric goal
        additional_context = f"""Your goal is to navigate to: {goal_description}

First, use the resolve_allocentric_goal tool to identify the target waypoint.
Then navigate to that waypoint using move_to.

Explain your spatial reasoning as you go."""

        # Use the existing multimodal run with this context
        return self.run_with_image(
            additional_context=additional_context,
            visual_feedback=visual_feedback
        )

    def _get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for prompts."""
        base_tools = create_tools(self.agent_state)
        allocentric_tools = create_allocentric_tools(self.allocentric, self.agent_state)
        all_tools = base_tools + allocentric_tools

        descriptions = ["Available tools:"]
        for tool in all_tools:
            descriptions.append(f"- {tool.name}: {tool.description.split(chr(10))[0]}")

        descriptions.append("\nTo use a tool, respond with:")
        descriptions.append("TOOL: <tool_name>")
        descriptions.append("ARGS: <json arguments>")

        return "\n".join(descriptions)

    def _parse_tool_call(self, text: str) -> tuple:
        """Parse tool name and args from response text."""
        import json

        tool_name = None
        tool_args = {}

        lines = text.split("\n")
        for line in lines:
            if line.startswith("TOOL:"):
                tool_name = line.replace("TOOL:", "").strip()
            elif line.startswith("ARGS:"):
                args_str = line.replace("ARGS:", "").strip()
                try:
                    tool_args = json.loads(args_str)
                except json.JSONDecodeError:
                    tool_args = {}

        return tool_name, tool_args

    def _execute_tool(self, tool_name: str, args: Dict) -> str:
        """Execute a tool by name (extended for allocentric tools)."""
        # First try parent implementation
        if tool_name in ["get_waypoints", "move_to", "rotate", "scale"]:
            return super()._execute_tool(tool_name, args)

        # Allocentric tools
        if tool_name == "get_spatial_relation":
            entity_a = args.get("entity_a", "")
            entity_b = args.get("entity_b", "")
            relation_types = args.get("relation_types", ["all"])

            a_id = self.allocentric.resolve_entity(entity_a)
            b_id = self.allocentric.resolve_entity(entity_b)

            if not a_id or not b_id:
                return f"Could not find entities: {entity_a}, {entity_b}"

            results = []
            a_name = self.allocentric.nodes[a_id].get('name', a_id)
            b_name = self.allocentric.nodes[b_id].get('name', b_id)

            if "all" in relation_types or "left_right" in relation_types:
                lr = self.allocentric.compute_left_right(a_id, b_id)
                results.append(f"{a_name} is {lr.replace('_', ' ')} {b_name}")

            if "all" in relation_types or "front_behind" in relation_types:
                fb = self.allocentric.compute_front_behind(a_id, b_id)
                results.append(f"{a_name} is {fb.replace('_', ' ')} {b_name}")

            return "\n".join(results)

        elif tool_name == "query_scene":
            query = args.get("query", "")
            # Simple query handling
            return f"Query: {query}\n(Use get_spatial_relation for specific entity pairs)"

        elif tool_name == "perspective_shift":
            ref_entity = args.get("reference_entity", "")
            direction = args.get("direction", None)
            return self.allocentric.describe_view_from(ref_entity)

        elif tool_name == "resolve_allocentric_goal":
            goal = args.get("goal_description", "")
            # Parse and resolve
            import re

            patterns = [
                (r'behind\s+(?:the\s+)?(\w+)', 'behind', 'front_behind'),
                (r'left of\s+(?:the\s+)?(\w+)', 'left_of', 'left_right'),
                (r'right of\s+(?:the\s+)?(\w+)', 'right_of', 'left_right'),
            ]

            for pattern, relation, rel_type in patterns:
                match = re.search(pattern, goal.lower())
                if match:
                    ref_name = match.group(1)
                    ref_id = self.allocentric.resolve_entity(ref_name)
                    if ref_id:
                        results = self.allocentric.find_entities_with_relation(
                            ref_id, relation, rel_type
                        )
                        if results:
                            target = results[0]
                            return f"Target resolved: {target['name']} ({target['id']})"

            return f"Could not resolve goal: {goal}"

        return f"Unknown tool: {tool_name}"


if __name__ == "__main__":
    main()
