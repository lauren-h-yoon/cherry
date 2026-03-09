#!/usr/bin/env python3
"""
embodied_agent.py - Embodied Spatial Intelligence Agent

An agent that uses primitive tools (output mechanisms) to express
spatial understanding. VLM sees image, reasons, and produces outputs
via tools - which are then evaluated against ground truth.

Supports multiple VLM backends:
- Claude (Anthropic API) - default
- Qwen VL (local via transformers/vLLM)
- OpenAI GPT-4o
- Ollama (local)
"""

import os
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from .primitive_tools import (
    create_primitive_tools,
    RecordedOutputs,
    EmbodiedState
)
from .ground_truth import GroundTruth
from .evaluator import SpatialEvaluator, EvaluationReport
from .tasks import SpatialTask, TaskType
from .model_providers import (
    VLMProvider,
    ModelResponse,
    ToolCall,
    create_model_provider,
    list_available_providers
)


EMBODIED_SYSTEM_PROMPT = """You are an embodied spatial reasoning agent analyzing a scene.

## Your Capabilities

You can express your spatial understanding using these tools:

1. **point_to(x, y, label)** - Point to a location in the image
   - Use when asked "where is X?" or "point to X"
   - x: horizontal position (0 = left edge)
   - y: vertical position (0 = top edge)

2. **draw_path(points, label)** - Draw a path through the scene
   - Use when asked to navigate, plan a route, or show how to move
   - points: list of (x, y) coordinates forming your path
   - First point = start, last point = destination

3. **mark_region(x1, y1, x2, y2, label)** - Mark a rectangular region
   - Use to highlight an object or area
   - Provide top-left (x1, y1) and bottom-right (x2, y2) corners

4. **move_to(x, y)** - Move to a position (for step-by-step navigation)
   - Use when executing a navigation sequence

5. **rotate(angle)** - Rotate your facing direction
   - Positive = clockwise, Negative = counter-clockwise

6. **look_at(x, y)** - Turn to face a location

## Image Coordinate System

- Origin (0, 0) is at the TOP-LEFT corner
- X increases to the RIGHT
- Y increases DOWNWARD
- Image dimensions are provided in the task

## Your Task

Look at the provided image and complete the spatial reasoning task.
Express your understanding by using the appropriate tools.

IMPORTANT:
- You must LOOK at the image to determine positions
- Estimate coordinates based on what you SEE in the image
- Use tools to express your spatial reasoning
- Be precise with coordinates based on visual inspection
"""


class EmbodiedSpatialAgent:
    """
    Agent for embodied spatial intelligence evaluation.

    The agent:
    1. Receives an image and a spatial task
    2. Uses primitive tools to express its understanding
    3. Outputs are recorded and evaluated against ground truth

    Supports multiple VLM backends:
    - Claude (Anthropic) - uses LangGraph ReAct agent
    - Qwen VL (local) - uses transformers/vLLM
    - OpenAI GPT-4o - uses OpenAI API
    - Ollama (local) - uses Ollama API
    """

    def __init__(
        self,
        provider: str = "claude",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        verbose: bool = True,
        use_vllm: bool = False,
        **provider_kwargs
    ):
        """
        Initialize the embodied spatial agent.

        Args:
            provider: VLM provider ('claude', 'qwen', 'openai', 'ollama')
            model_name: Specific model name (uses provider default if not specified)
            api_key: API key (for API-based providers)
            temperature: Model temperature
            verbose: Print agent outputs
            use_vllm: Use vLLM for local inference (Qwen only)
            **provider_kwargs: Additional provider-specific arguments
        """
        self.provider_name = provider
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose

        # Build provider kwargs
        kwargs = {
            "temperature": temperature,
            "verbose": verbose,
            **provider_kwargs
        }

        if api_key:
            kwargs["api_key"] = api_key

        if use_vllm:
            kwargs["use_vllm"] = True

        # Create the provider
        self.provider: VLMProvider = create_model_provider(
            provider,
            model_name=model_name,
            **kwargs
        )

        # For Claude, we can still use LangGraph for better tool handling
        self.use_langgraph = provider.lower() in ["claude", "claude-sonnet", "anthropic"]
        self.llm = None
        self.agent_graph = None

        if self.use_langgraph:
            self._setup_langgraph()

        # Will be initialized per task
        self.recorded_outputs: Optional[RecordedOutputs] = None
        self.embodied_state: Optional[EmbodiedState] = None
        self.tools: Optional[List] = None
        self.tool_map: Dict[str, Any] = {}

    def _setup_langgraph(self):
        """Set up LangGraph for Claude provider."""
        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.llm = ChatAnthropic(
            model=self.provider.model_name,
            anthropic_api_key=api_key,
            temperature=self.temperature,
            max_tokens=4096
        )

    def _setup_tools(self, initial_position: tuple = (0, 0)):
        """Set up fresh tools for a new task."""
        self.recorded_outputs = RecordedOutputs()
        self.embodied_state = EmbodiedState(position=initial_position)

        self.tools, self.recorded_outputs, self.embodied_state = create_primitive_tools(
            self.recorded_outputs,
            self.embodied_state
        )

        # Build tool map for manual execution
        self.tool_map = {tool.name: tool for tool in self.tools}

        # Set up LangGraph agent if using Claude
        if self.use_langgraph:
            from langgraph.prebuilt import create_react_agent
            self.agent_graph = create_react_agent(
                self.llm,
                self.tools,
                prompt=EMBODIED_SYSTEM_PROMPT
            )

    def _get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions in a standard format."""
        tool_defs = []
        for tool in self.tools:
            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }

            # Extract parameters from tool schema
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.schema()
                tool_def["parameters"]["properties"] = schema.get("properties", {})
                tool_def["parameters"]["required"] = schema.get("required", [])

            tool_defs.append(tool_def)

        return tool_defs

    def _execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[str]:
        """Execute tool calls and return results."""
        results = []
        for tc in tool_calls:
            if tc.name in self.tool_map:
                tool = self.tool_map[tc.name]
                try:
                    result = tool.invoke(tc.arguments)
                    results.append(f"{tc.name}: {result}")
                    if self.verbose:
                        print(f"  Tool: {tc.name}({tc.arguments}) -> {result}")
                except Exception as e:
                    results.append(f"{tc.name}: Error - {str(e)}")
            else:
                results.append(f"{tc.name}: Unknown tool")
        return results

    def _encode_image(self, image_path: str, max_size_bytes: int = 3_500_000) -> tuple:
        """Encode image as base64, resizing if needed to stay under size limit.

        Args:
            image_path: Path to image file
            max_size_bytes: Maximum size in bytes (default 3.5MB - accounts for ~33% base64 overhead to stay under 5MB API limit)

        Returns:
            Tuple of (base64_data, media_type)
        """
        from PIL import Image
        import io

        # Detect original media type
        ext = Path(image_path).suffix.lower()
        if ext in (".jpg", ".jpeg"):
            original_media_type = "image/jpeg"
        elif ext == ".png":
            original_media_type = "image/png"
        elif ext == ".gif":
            original_media_type = "image/gif"
        elif ext == ".webp":
            original_media_type = "image/webp"
        else:
            original_media_type = "image/jpeg"

        # First check if original is small enough
        with open(image_path, "rb") as f:
            original_data = f.read()

        if len(original_data) <= max_size_bytes:
            return base64.standard_b64encode(original_data).decode("utf-8"), original_media_type

        # Need to resize - will convert to JPEG for smaller size
        img = Image.open(image_path)

        # Calculate scale factor to get under size limit
        # Start with 80% and keep reducing until small enough
        scale = 0.8
        while scale > 0.1:
            new_size = (int(img.width * scale), int(img.height * scale))
            resized = img.resize(new_size, Image.LANCZOS)

            buf = io.BytesIO()
            # Use JPEG for smaller size
            if img.mode in ('RGBA', 'P'):
                resized = resized.convert('RGB')
            resized.save(buf, format='JPEG', quality=85)

            if buf.tell() <= max_size_bytes:
                return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"

            scale *= 0.8

        # Final fallback: very small
        resized = img.resize((800, 600), Image.LANCZOS)
        buf = io.BytesIO()
        if img.mode in ('RGBA', 'P'):
            resized = resized.convert('RGB')
        resized.save(buf, format='JPEG', quality=70)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"

    def run_task(
        self,
        task: SpatialTask,
        image_path: str,
        image_size: tuple = None
    ) -> Dict[str, Any]:
        """
        Run a spatial task and return results.

        Args:
            task: SpatialTask to execute
            image_path: Path to scene image
            image_size: (height, width) of image

        Returns:
            Dict with task results and recorded outputs
        """
        # Setup fresh tools
        initial_pos = task.expected_targets.get('start_position', (0, 0))
        if isinstance(initial_pos, (list, tuple)):
            self._setup_tools(initial_position=tuple(initial_pos))
        else:
            self._setup_tools()

        # Build prompt with image size info
        size_info = ""
        if image_size:
            size_info = f"\n\nImage dimensions: {image_size[1]}w x {image_size[0]}h pixels"

        full_prompt = task.prompt + size_info

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task.instruction}")
            print(f"Type: {task.task_type.value}")
            print(f"Model: {self.provider.model_name}")
            print(f"{'='*60}")

        # Run with appropriate method based on provider
        if self.use_langgraph:
            response_text = self._run_with_langgraph(image_path, full_prompt)
        else:
            response_text = self._run_with_provider(image_path, full_prompt)

        if self.verbose:
            print(f"\nAgent Response:\n{response_text[:500]}...")
            print(f"\nRecorded Outputs:")
            print(f"  Paths: {len(self.recorded_outputs.paths)}")
            print(f"  Points: {len(self.recorded_outputs.points)}")
            print(f"  Regions: {len(self.recorded_outputs.regions)}")
            print(f"  Movements: {len(self.recorded_outputs.movements)}")

        return {
            'task': task.to_dict(),
            'response': response_text,
            'recorded_outputs': self.recorded_outputs.to_dict(),
            'embodied_state': {
                'position': self.embodied_state.position,
                'facing_angle': self.embodied_state.facing_angle
            },
            'model': self.provider.model_name,
            'provider': self.provider_name
        }

    def _run_with_langgraph(self, image_path: str, prompt: str) -> str:
        """Run task using LangGraph ReAct agent (Claude)."""
        from langchain_core.messages import HumanMessage

        image_b64, media_type = self._encode_image(image_path)

        message = HumanMessage(
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )

        result = self.agent_graph.invoke({"messages": [message]})

        # Extract all text content from AI messages
        response_parts = []
        for msg in result.get("messages", []):
            if not hasattr(msg, "content"):
                continue

            # Skip non-AI messages
            msg_type = getattr(msg, "type", "")
            if msg_type not in ("ai", "AIMessage", ""):
                continue

            content = msg.content
            if isinstance(content, str):
                if content.strip():
                    response_parts.append(content)
            elif isinstance(content, list):
                # Content is a list of blocks (text, tool_use, etc.)
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text.strip():
                            response_parts.append(text)
                    elif isinstance(block, str) and block.strip():
                        response_parts.append(block)

        return "\n".join(response_parts) if response_parts else ""

    def _run_with_provider(self, image_path: str, prompt: str) -> str:
        """Run task using generic provider (Qwen, OpenAI, Ollama)."""
        tool_defs = self._get_tool_definitions()

        # Generate response
        response = self.provider.generate(
            image_path=image_path,
            prompt=prompt,
            system_prompt=EMBODIED_SYSTEM_PROMPT,
            tools=tool_defs
        )

        # Execute any tool calls
        if response.tool_calls:
            if self.verbose:
                print(f"\nExecuting {len(response.tool_calls)} tool calls:")
            self._execute_tool_calls(response.tool_calls)

        return response.text

    def _scale_qwen_coordinates(self, image_size: tuple):
        """
        Scale coordinates from Qwen's normalized 0-1000 range to pixel coordinates.

        Qwen VL models output coordinates in a 0-1000 normalized space.
        This method converts them to actual pixel coordinates.

        Args:
            image_size: (height, width) of the original image
        """
        # Only apply to Qwen/vLLM providers
        if self.provider_name.lower() not in ['qwen', 'vllm', 'qwen-server', 'qwen3-vl']:
            return

        height, width = image_size
        scale_x = width / 1000.0
        scale_y = height / 1000.0

        # Scale points
        for point in self.recorded_outputs.points:
            point['x'] = int(point['x'] * scale_x)
            point['y'] = int(point['y'] * scale_y)

        # Scale paths
        for path in self.recorded_outputs.paths:
            scaled_points = []
            for p in path['points']:
                if isinstance(p, (list, tuple)):
                    scaled_points.append((int(p[0] * scale_x), int(p[1] * scale_y)))
                else:
                    scaled_points.append(p)
            path['points'] = scaled_points
            if path.get('start'):
                path['start'] = (int(path['start'][0] * scale_x), int(path['start'][1] * scale_y))
            if path.get('end'):
                path['end'] = (int(path['end'][0] * scale_x), int(path['end'][1] * scale_y))

        # Scale regions
        for region in self.recorded_outputs.regions:
            if 'bbox' in region:
                bbox = region['bbox']
                region['bbox'] = [
                    int(bbox[0] * scale_x),
                    int(bbox[1] * scale_y),
                    int(bbox[2] * scale_x),
                    int(bbox[3] * scale_y)
                ]
            if 'center' in region:
                region['center'] = (
                    int(region['center'][0] * scale_x),
                    int(region['center'][1] * scale_y)
                )

        # Scale gaze targets
        for gaze in self.recorded_outputs.gaze:
            if 'target' in gaze:
                gaze['target'] = (
                    int(gaze['target'][0] * scale_x),
                    int(gaze['target'][1] * scale_y)
                )

    def run_and_evaluate(
        self,
        task: SpatialTask,
        image_path: str,
        ground_truth: GroundTruth
    ) -> tuple:
        """
        Run a task and evaluate against ground truth.

        Args:
            task: SpatialTask to execute
            image_path: Path to scene image
            ground_truth: GroundTruth for evaluation

        Returns:
            Tuple of (EvaluationReport, recorded_outputs_dict, response_text)
        """
        # Run the task
        result = self.run_task(
            task,
            image_path,
            image_size=ground_truth.image_size
        )

        # Scale coordinates for Qwen VL models (0-1000 -> pixel coords)
        self._scale_qwen_coordinates(ground_truth.image_size)

        # Capture recorded outputs before evaluation
        recorded_outputs_dict = self.recorded_outputs.to_dict()
        response_text = result.get('response', '')

        # Evaluate (including yes/no answer extraction for QA tasks)
        evaluator = SpatialEvaluator(ground_truth)
        report = evaluator.evaluate_with_response(
            self.recorded_outputs,
            response_text,
            task_id=task.task_id,
            task_type=task.task_type.value,
            expected_targets=task.expected_targets
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print("EVALUATION RESULTS")
            print(f"{'='*60}")
            print(f"Overall Score: {report.overall_score:.2%}")

            if report.pointing_results:
                for pr in report.pointing_results:
                    if pr.correct:
                        status = "✓"
                        extra = ""
                    elif pr.hit_same_category:
                        status = "◐"  # Partial credit symbol
                        extra = f" (hit {pr.hit_object} instead)"
                    else:
                        status = "✗"
                        extra = ""
                    print(f"  {status} Pointing to {pr.target_object}: "
                          f"error={pr.distance_error:.0f}px, score={pr.score:.0%}{extra}")

            if report.path_results:
                for pr in report.path_results:
                    print(f"  Path: start={'✓' if pr.start_correct else '✗'}, "
                          f"end={'✓' if pr.end_correct else '✗'}, "
                          f"valid={'✓' if pr.path_valid else '✗'}, "
                          f"efficiency={pr.efficiency:.1%}")
                    if pr.collisions:
                        print(f"    Collisions: {', '.join(pr.collisions)}")

            if report.yes_no_results:
                for ynr in report.yes_no_results:
                    if ynr.answer_found:
                        status = "✓" if ynr.correct else "✗"
                        vlm_ans = "YES" if ynr.vlm_answer else "NO"
                        gt_ans = "YES" if ynr.gt_answer else "NO"
                        print(f"  {status} Yes/No Answer: VLM={vlm_ans}, GT={gt_ans}")
                    else:
                        print(f"  ? Yes/No Answer: Not found in response")

        return report, recorded_outputs_dict, response_text


class EmbodiedEvaluationRunner:
    """
    Runs batch evaluations of embodied spatial intelligence.

    Supports multiple VLM backends for comparative evaluation.
    """

    def __init__(
        self,
        spatial_graph_path: str,
        image_path: str,
        provider: str = "claude",
        model_name: Optional[str] = None,
        verbose: bool = True,
        **provider_kwargs
    ):
        """
        Initialize evaluation runner.

        Args:
            spatial_graph_path: Path to spatial_graph.json
            image_path: Path to scene image
            provider: VLM provider ('claude', 'qwen', 'openai', 'ollama')
            model_name: Specific model name (uses provider default if not specified)
            verbose: Print progress
            **provider_kwargs: Additional provider-specific arguments
        """
        self.ground_truth = GroundTruth(spatial_graph_path)
        self.image_path = image_path
        self.provider = provider
        self.model_name = model_name
        self.verbose = verbose

        self.agent = EmbodiedSpatialAgent(
            provider=provider,
            model_name=model_name,
            verbose=verbose,
            **provider_kwargs
        )

    def run_single_task(self, task: SpatialTask) -> tuple:
        """Run a single task and return evaluation with outputs.

        For allocentric tasks, uses the annotated image with viewpoint markers
        if available. Otherwise uses the original image.
        """
        import tempfile
        from PIL import Image

        # Use annotated image for allocentric tasks if available
        image_to_use = self.image_path
        temp_file = None

        if hasattr(task, 'annotated_image') and task.annotated_image is not None:
            # If it's a PIL Image, save to temp file
            if isinstance(task.annotated_image, Image.Image):
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                task.annotated_image.convert('RGB').save(temp_file.name, 'JPEG', quality=95)
                image_to_use = temp_file.name
                if self.verbose:
                    print(f"Using annotated image with viewpoint markers")
            elif isinstance(task.annotated_image, str):
                image_to_use = task.annotated_image
                if self.verbose:
                    print(f"Using annotated image: {image_to_use}")

        try:
            return self.agent.run_and_evaluate(
                task,
                image_to_use,
                self.ground_truth
            )
        finally:
            # Clean up temp file
            if temp_file is not None:
                import os
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    def run_task_suite(
        self,
        tasks: List[SpatialTask]
    ) -> Dict[str, Any]:
        """
        Run a suite of tasks and aggregate results.

        Args:
            tasks: List of tasks to run

        Returns:
            Dict with per-task and aggregate results
        """
        results = {
            'model': {
                'provider': self.provider,
                'model_name': self.agent.provider.model_name
            },
            'image_path': self.image_path,
            'image_size': self.ground_truth.image_size,
            'tasks': [],
            'by_type': {},
            'overall': {
                'total_tasks': len(tasks),
                'total_score': 0.0
            }
        }

        for i, task in enumerate(tasks):
            if self.verbose:
                print(f"\n{'#'*60}")
                print(f"# Task {i+1}/{len(tasks)}: {task.task_type.value}")
                print(f"{'#'*60}")

            report, recorded_outputs, response_text = self.run_single_task(task)

            task_result = {
                'task_id': task.task_id,
                'task_type': task.task_type.value,
                'instruction': task.instruction,
                'score': report.overall_score,
                'report': report.to_dict(),
                'model_response': response_text,
                'recorded_outputs': recorded_outputs,
                'expected_targets': task.expected_targets
            }
            results['tasks'].append(task_result)

            # Aggregate by type
            task_type = task.task_type.value
            if task_type not in results['by_type']:
                results['by_type'][task_type] = {
                    'count': 0,
                    'total_score': 0.0,
                    'scores': []
                }
            results['by_type'][task_type]['count'] += 1
            results['by_type'][task_type]['total_score'] += report.overall_score
            results['by_type'][task_type]['scores'].append(report.overall_score)

        # Compute aggregates
        if tasks:
            results['overall']['total_score'] = sum(
                t['score'] for t in results['tasks']
            ) / len(tasks)

        for task_type, data in results['by_type'].items():
            if data['count'] > 0:
                data['average_score'] = data['total_score'] / data['count']

        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        # Print model info
        if 'model' in results:
            print(f"\nModel: {results['model'].get('model_name', 'unknown')}")
            print(f"Provider: {results['model'].get('provider', 'unknown')}")

        print(f"\nOverall Score: {results['overall']['total_score']:.1%}")
        print(f"Total Tasks: {results['overall']['total_tasks']}")

        print("\nBy Task Type:")
        for task_type, data in results['by_type'].items():
            print(f"  {task_type}:")
            print(f"    Count: {data['count']}")
            print(f"    Average: {data.get('average_score', 0):.1%}")

        print("\n" + "="*60)
