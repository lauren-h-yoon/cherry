#!/usr/bin/env python3
"""
visualize_tasks.py - Unified 3-panel task visualization tool

Three panels:
1. INPUT (Left): Image with markers + prompt
2. GROUND TRUTH (Middle): Expected answer with tolerance regions
3. OUTPUT (Right): Model's response (only shown after evaluation)

Modes:
- Sanity Check (--sanity): Shows INPUT + GROUND TRUTH panels
- Full Visualization (--results): Shows all 3 panels after evaluation

Usage:
    # Sanity check - verify inputs and ground truth for all 4 task types
    python visualize_tasks.py --sanity --graph spatial_outputs/graph.json --image scene.jpg

    # Full visualization after evaluation
    python visualize_tasks.py --results eval_results/results.json --image scene.jpg --graph spatial_outputs/graph.json
"""

import json
import argparse
import textwrap
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from spatial_agent.ground_truth import GroundTruth
from spatial_agent.tasks import TaskGenerator, TaskType
from spatial_agent.viewpoint_annotator import ViewpointAnnotator


class TaskVisualizer:
    """Unified 3-panel visualizer for spatial tasks."""

    def __init__(self, image_path: str, graph_path: Optional[str] = None):
        self.image_path = image_path
        self.base_image = Image.open(image_path).convert('RGB')

        # Load ground truth if provided
        self.gt = None
        self.graph_data = None
        if graph_path:
            self.gt = GroundTruth(graph_path)
            with open(graph_path) as f:
                self.graph_data = json.load(f)

        # Load fonts
        try:
            self.font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            self.font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
            self.font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            try:
                self.font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
                self.font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
                self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                self.font_large = ImageFont.load_default()
                self.font_medium = self.font_large
                self.font_small = self.font_large

    def wrap_text(self, text: str, width: int = 50, max_lines: int = 10) -> str:
        """Wrap text to specified character width."""
        lines = []
        for para in text.split('\n'):
            if para.strip():
                lines.extend(textwrap.wrap(para, width=width))
            else:
                lines.append('')
        return '\n'.join(lines[:max_lines])

    def get_object_bbox(self, object_name: str) -> Optional[List[float]]:
        """Get bounding box for an object from ground truth."""
        if not self.graph_data:
            return None

        for node in self.graph_data.get('nodes', []):
            # Match by name or category
            if node.get('name') == object_name or object_name.startswith(node.get('category', '')):
                return node.get('bbox')
        return None

    def get_object_center(self, object_name: str) -> Optional[Tuple[float, float]]:
        """Get center position for an object."""
        if not self.graph_data:
            return None

        for node in self.graph_data.get('nodes', []):
            if node.get('name') == object_name or object_name.startswith(node.get('category', '')):
                center = node.get('bbox_center')
                if center:
                    return (center[0], center[1])
        return None

    def draw_tolerance_region(self, img: Image.Image, bbox: List[float],
                              color: Tuple[int, int, int] = (0, 255, 0),
                              label: str = None) -> Image.Image:
        """Draw tolerance region (bounding box) on image."""
        img = img.copy()
        draw = ImageDraw.Draw(img, 'RGBA')

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=(*color, 50), outline=(*color, 200), width=4)

        # Dashed border effect (draw multiple small lines)
        # Just use solid for simplicity

        # Label
        if label:
            draw.text((x1, y1 - 25), label, fill=(*color, 255), font=self.font_small)

        return img

    def draw_facing_arrow(self, img: Image.Image, position: Tuple[float, float],
                          facing: str, color: Tuple[int, int, int] = (0, 200, 255),
                          arrow_length: int = 150, label: str = None) -> Image.Image:
        """Draw a facing direction arrow from the given position.

        The arrow identifies both the anchor object and its facing direction.
        """
        img = img.copy()
        draw = ImageDraw.Draw(img)

        x, y = int(position[0]), int(position[1])

        # Calculate arrow endpoint based on facing direction
        if facing == "camera":
            end_x, end_y = x, y - arrow_length  # Up (toward camera)
        elif facing == "away":
            end_x, end_y = x, y + arrow_length  # Down (away from camera)
        elif facing == "left":
            end_x, end_y = x - arrow_length, y  # Left
        elif facing == "right":
            end_x, end_y = x + arrow_length, y  # Right
        else:
            end_x, end_y = x, y - arrow_length  # Default to camera

        # Draw circle at origin to mark the anchor point
        draw.ellipse([x-15, y-15, x+15, y+15], fill=color, outline=(255, 255, 255), width=3)

        # Draw arrow line
        draw.line([(x, y), (end_x, end_y)], fill=color, width=8)

        # Draw arrow head
        angle = math.atan2(end_y - y, end_x - x)
        head_size = 30
        draw.polygon([
            (end_x, end_y),
            (end_x - head_size * math.cos(angle - 0.5), end_y - head_size * math.sin(angle - 0.5)),
            (end_x - head_size * math.cos(angle + 0.5), end_y - head_size * math.sin(angle + 0.5))
        ], fill=color)

        # Label showing anchor object and facing direction
        if label:
            label_text = f"{label} (facing {facing})"
        else:
            label_text = f"facing {facing}"

        # Position label to avoid overlap with arrow
        if facing == "camera":
            label_x, label_y = x + 25, y + 20
        elif facing == "away":
            label_x, label_y = x + 25, y - 40
        else:
            label_x, label_y = x - 50, y - 50

        draw.text((label_x, label_y), label_text, fill=color, font=self.font_small)

        return img

    def draw_object_marker(self, img: Image.Image, position: Tuple[float, float],
                           color: Tuple[int, int, int] = (255, 0, 0),
                           label: str = None, marker_type: str = "crosshair") -> Image.Image:
        """Draw object marker on image."""
        img = img.copy()
        draw = ImageDraw.Draw(img)

        x, y = int(position[0]), int(position[1])

        if marker_type == "crosshair":
            # Crosshair
            draw.line([(x-30, y), (x+30, y)], fill=color, width=4)
            draw.line([(x, y-30), (x, y+30)], fill=color, width=4)
            draw.ellipse([x-20, y-20, x+20, y+20], outline=color, width=3)
        elif marker_type == "circle":
            draw.ellipse([x-25, y-25, x+25, y+25], outline=color, width=4)
        elif marker_type == "square":
            draw.rectangle([x-20, y-20, x+20, y+20], outline=color, width=4)

        if label:
            draw.text((x + 30, y - 10), label, fill=color, font=self.font_small)

        return img

    def draw_path_on_image(self, img: Image.Image, points: List,
                           color: Tuple[int, int, int] = (138, 43, 226)) -> Image.Image:
        """Draw path with waypoints on image."""
        img = img.copy()
        draw = ImageDraw.Draw(img)

        if len(points) < 2:
            return img

        # Draw lines
        for i in range(len(points) - 1):
            p1 = tuple(points[i]) if isinstance(points[i], list) else points[i]
            p2 = tuple(points[i+1]) if isinstance(points[i+1], list) else points[i+1]
            draw.line([p1, p2], fill=color, width=6)

        # Draw waypoints
        for i, p in enumerate(points):
            x, y = p if isinstance(p, (list, tuple)) else (p['x'], p['y'])
            x, y = int(x), int(y)
            r = 15
            if i == 0:
                draw.ellipse([x-r, y-r, x+r, y+r], fill=(0, 150, 255), outline=(255,255,255), width=2)
            elif i == len(points) - 1:
                draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 150, 0), outline=(255,255,255), width=2)
            else:
                draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=(255,255,255), width=2)

        return img

    def create_input_panel(
        self,
        task_type: str,
        input_image: Image.Image,
        prompt: str,
        instruction: str,
        panel_width: int = 640,
        panel_height: int = 800
    ) -> Image.Image:
        """Create INPUT panel (left) showing image and prompt."""
        panel = Image.new('RGB', (panel_width, panel_height), (25, 25, 25))
        draw = ImageDraw.Draw(panel)

        # Title
        draw.text((15, 8), "INPUT", fill=(100, 200, 100), font=self.font_large)

        # Scale and paste image
        img_area_height = panel_height - 280
        scale = min((panel_width - 30) / input_image.width, img_area_height / input_image.height)
        new_w, new_h = int(input_image.width * scale), int(input_image.height * scale)
        scaled_img = input_image.resize((new_w, new_h), Image.LANCZOS)

        img_x = (panel_width - new_w) // 2
        panel.paste(scaled_img, (img_x, 45))

        # Task instruction
        y = 50 + new_h + 10
        draw.text((15, y), f"Task: {instruction[:60]}", fill=(255, 220, 100), font=self.font_medium)

        # Prompt
        y += 30
        draw.text((15, y), "Prompt:", fill=(150, 200, 255), font=self.font_medium)
        y += 25
        wrapped_prompt = self.wrap_text(prompt, width=int(panel_width / 9), max_lines=12)
        draw.text((15, y), wrapped_prompt, fill=(200, 200, 200), font=self.font_small)

        return panel

    def create_ground_truth_panel(
        self,
        task_type: str,
        base_image: Image.Image,
        expected_targets: Dict,
        panel_width: int = 640,
        panel_height: int = 800
    ) -> Image.Image:
        """Create GROUND TRUTH panel (middle) showing expected answer."""
        panel = Image.new('RGB', (panel_width, panel_height), (30, 30, 35))
        draw = ImageDraw.Draw(panel)

        # Title
        draw.text((15, 8), "GROUND TRUTH", fill=(255, 200, 100), font=self.font_large)

        # Create annotated ground truth image
        gt_img = base_image.copy()

        if task_type == "object_localization":
            # Show target object with tolerance region (bbox)
            target_obj = expected_targets.get('target_object', '')
            bbox = self.get_object_bbox(target_obj)
            center = self.get_object_center(target_obj)

            if bbox:
                gt_img = self.draw_tolerance_region(gt_img, bbox, color=(0, 255, 0),
                                                    label=f"TARGET: {target_obj}")
            if center:
                gt_img = self.draw_object_marker(gt_img, center, color=(0, 255, 0), marker_type="crosshair")

        elif task_type == "path_planning":
            # Show start position and target object bbox as destination
            start_pos = expected_targets.get('start_position')
            goal_obj = expected_targets.get('goal_object', '')
            goal_bbox = self.get_object_bbox(goal_obj)
            goal_center = self.get_object_center(goal_obj)

            # Draw start marker and start tolerance region (blue)
            if start_pos:
                annotator = ViewpointAnnotator(self.image_path)
                annotator.add_start_marker((int(start_pos[0]), int(start_pos[1])), "START")
                gt_img = annotator.get_image().convert('RGB')

                # Draw start tolerance region (100px radius around start)
                start_radius = 100
                start_bbox = [
                    start_pos[0] - start_radius,
                    start_pos[1] - start_radius,
                    start_pos[0] + start_radius,
                    start_pos[1] + start_radius
                ]
                gt_img = self.draw_tolerance_region(gt_img, start_bbox, color=(0, 150, 255),
                                                    label="START ZONE")

            # Draw target tolerance region (green)
            if goal_bbox:
                gt_img = self.draw_tolerance_region(gt_img, goal_bbox, color=(0, 255, 0),
                                                    label=f"GOAL: {goal_obj}")
            elif goal_center:
                gt_img = self.draw_object_marker(gt_img, goal_center, color=(0, 255, 0),
                                                 label=f"GOAL: {goal_obj}")

        elif task_type == "egocentric_spatial_qa":
            # Show target object and expected YES/NO answer
            target_obj = expected_targets.get('target_object', '')
            gt_answer = expected_targets.get('ground_truth_answer')
            direction = expected_targets.get('direction', '')

            center = self.get_object_center(target_obj)
            bbox = self.get_object_bbox(target_obj)

            if bbox:
                gt_img = self.draw_tolerance_region(gt_img, bbox, color=(0, 255, 0),
                                                    label=f"TARGET: {target_obj}")
            if center:
                gt_img = self.draw_object_marker(gt_img, center, color=(0, 255, 0))

        elif task_type == "allocentric_spatial_qa":
            # Reference object is the anchor - just draw facing arrow on it
            # No blue box needed - the arrow identifies the anchor
            target_obj = expected_targets.get('target_object', '')
            ref_obj = expected_targets.get('reference_object', '')
            relation = expected_targets.get('relation', '')
            facing = expected_targets.get('facing', 'camera')
            gt_answer = expected_targets.get('ground_truth_answer')

            # Get positions
            ref_center = self.get_object_center(ref_obj)
            target_bbox = self.get_object_bbox(target_obj)

            # Draw facing arrow FROM the reference object (identifies anchor + direction)
            if ref_center:
                gt_img = self.draw_facing_arrow(gt_img, ref_center, facing, color=(0, 200, 255),
                                                label=ref_obj)

            # Draw target object (green) - this is what model needs to point to
            if target_bbox:
                gt_img = self.draw_tolerance_region(gt_img, target_bbox, color=(0, 255, 0),
                                                    label=f"TARGET: {target_obj}")

        # Scale and paste ground truth image
        img_area_height = panel_height - 200
        scale = min((panel_width - 30) / gt_img.width, img_area_height / gt_img.height)
        new_w, new_h = int(gt_img.width * scale), int(gt_img.height * scale)
        scaled_img = gt_img.resize((new_w, new_h), Image.LANCZOS)

        img_x = (panel_width - new_w) // 2
        panel.paste(scaled_img, (img_x, 45))

        # Expected answer text
        y = 50 + new_h + 15
        draw.text((15, y), "Expected Answer:", fill=(150, 200, 255), font=self.font_medium)
        y += 28

        if task_type == "object_localization":
            target = expected_targets.get('target_descriptor', expected_targets.get('target_object', ''))
            draw.text((15, y), f"Point to: {target}", fill=(200, 200, 200), font=self.font_small)
            y += 22
            draw.text((15, y), "Tolerance: Within object bounding box", fill=(150, 150, 150), font=self.font_small)

        elif task_type == "path_planning":
            goal = expected_targets.get('goal_descriptor', expected_targets.get('goal_object', ''))
            draw.text((15, y), f"Draw path from START to: {goal}", fill=(200, 200, 200), font=self.font_small)
            y += 25
            draw.text((15, y), "Evaluation criteria:", fill=(150, 200, 255), font=self.font_medium)
            y += 22
            draw.text((15, y), "1. Path endpoint within goal bbox (green)", fill=(100, 255, 100), font=self.font_small)
            y += 18
            draw.text((15, y), "2. Path starts near START marker (blue)", fill=(100, 180, 255), font=self.font_small)
            y += 18
            draw.text((15, y), "3. Waypoints form a reasonable route", fill=(180, 180, 180), font=self.font_small)

        elif task_type == "egocentric_spatial_qa":
            gt_answer = expected_targets.get('ground_truth_answer')
            answer_str = "YES" if gt_answer else "NO"
            answer_color = (100, 255, 100) if gt_answer else (255, 100, 100)
            draw.text((15, y), f"Correct Answer: {answer_str}", fill=answer_color, font=self.font_medium)
            y += 28
            direction = expected_targets.get('direction', '')
            draw.text((15, y), f"Direction checked: {direction}", fill=(150, 150, 150), font=self.font_small)

        elif task_type == "allocentric_spatial_qa":
            # Reference object is anchor (with arrow) - model must point to BOTH objects
            ref = expected_targets.get('reference_object', '')
            target = expected_targets.get('target_object', '')
            relation = expected_targets.get('relation', '')
            facing = expected_targets.get('facing', '')
            gt_answer = expected_targets.get('ground_truth_answer')

            # Format relation phrase
            relation_phrase = {
                'left': 'to their left',
                'right': 'to their right',
                'in_front': 'in front of them',
                'behind': 'behind them'
            }.get(relation, relation)

            draw.text((15, y), "Required outputs:", fill=(150, 200, 255), font=self.font_medium)
            y += 25
            draw.text((15, y), f"1. Point to {ref} (arrow shows direction)", fill=(100, 200, 255), font=self.font_small)
            y += 20
            draw.text((15, y), f"2. Point to {target} (green box)", fill=(100, 255, 100), font=self.font_small)
            y += 25

            draw.text((15, y), f"Arrow: {ref} facing {facing}", fill=(100, 200, 255), font=self.font_small)
            y += 20
            draw.text((15, y), f"Q: Is {target} {relation_phrase}?", fill=(200, 200, 200), font=self.font_small)
            y += 25

            answer_str = "YES" if gt_answer else "NO"
            answer_color = (100, 255, 100) if gt_answer else (255, 100, 100)
            draw.text((15, y), f"Correct Answer: {answer_str}", fill=answer_color, font=self.font_medium)

        return panel

    def create_output_panel(
        self,
        output_image: Image.Image,
        response: str,
        outputs: Dict,
        score: float,
        ground_truth: Dict,
        panel_width: int = 640,
        panel_height: int = 800
    ) -> Image.Image:
        """Create OUTPUT panel (right) showing model response."""
        panel = Image.new('RGB', (panel_width, panel_height), (35, 25, 25))
        draw = ImageDraw.Draw(panel)

        # Title with score
        score_color = (100, 220, 100) if score >= 0.5 else (220, 100, 100)
        draw.text((15, 8), f"MODEL OUTPUT (Score: {score:.0%})", fill=score_color, font=self.font_large)

        # Draw model outputs on image
        out_img = output_image.copy()

        # Draw paths
        for path in outputs.get('paths', []):
            points = path.get('points', [])
            out_img = self.draw_path_on_image(out_img, points)

        # Draw points
        for point in outputs.get('points', []):
            x, y = point['x'], point['y']
            out_img = self.draw_object_marker(out_img, (x, y), color=(255, 0, 0),
                                              label=point.get('label', ''), marker_type="crosshair")

        # Scale and paste image
        img_area_height = panel_height - 200
        scale = min((panel_width - 30) / out_img.width, img_area_height / out_img.height)
        new_w, new_h = int(out_img.width * scale), int(out_img.height * scale)
        scaled_img = out_img.resize((new_w, new_h), Image.LANCZOS)

        img_x = (panel_width - new_w) // 2
        panel.paste(scaled_img, (img_x, 45))

        # Model response
        y = 50 + new_h + 15
        draw.text((15, y), "Model Response:", fill=(150, 200, 255), font=self.font_medium)
        y += 28
        response_text = response if response else "(no text response)"
        wrapped = self.wrap_text(response_text, width=int(panel_width / 9), max_lines=4)
        draw.text((15, y), wrapped, fill=(200, 200, 200), font=self.font_small)

        # Tool outputs
        y += 80
        draw.text((15, y), "Tool Outputs:", fill=(150, 200, 255), font=self.font_medium)
        y += 25

        out_info = []
        if outputs.get('paths'):
            path = outputs['paths'][0]
            pts = path.get('points', [])
            out_info.append(f"Path: {len(pts)} waypoints")
            if pts:
                out_info.append(f"  End: {pts[-1]}")
        if outputs.get('points'):
            for pt in outputs['points'][:2]:
                out_info.append(f"Point: ({pt['x']}, {pt['y']})")
        if not out_info:
            out_info.append("No tool outputs recorded")

        draw.text((15, y), "\n".join(out_info), fill=(180, 180, 180), font=self.font_small)

        return panel

    def visualize_sanity_check(
        self,
        tasks: List,
        output_dir: str
    ):
        """Generate sanity check images (INPUT + GROUND TRUTH) for all tasks."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, task in enumerate(tasks):
            task_type = task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type)

            # Get input image
            if hasattr(task, 'annotated_image') and task.annotated_image is not None:
                if isinstance(task.annotated_image, Image.Image):
                    input_img = task.annotated_image.convert('RGB')
                else:
                    input_img = Image.open(task.annotated_image).convert('RGB')
            else:
                input_img = self.base_image.copy()

            # Create panels
            input_panel = self.create_input_panel(
                task_type=task_type,
                input_image=input_img,
                prompt=task.prompt,
                instruction=task.instruction
            )

            gt_panel = self.create_ground_truth_panel(
                task_type=task_type,
                base_image=self.base_image,
                expected_targets=task.expected_targets
            )

            # Combine 2 panels for sanity check
            combined = Image.new('RGB', (input_panel.width + gt_panel.width + 15, input_panel.height), (15, 15, 15))
            combined.paste(input_panel, (0, 0))
            combined.paste(gt_panel, (input_panel.width + 15, 0))

            # Save
            filename = f"task_{i+1:02d}_{task_type}.png"
            combined.save(output_path / filename)
            print(f"Created: {filename}")

        print(f"\nSanity check images saved to: {output_dir}")

    def visualize_results(
        self,
        results: Dict,
        output_dir: str
    ):
        """Generate full 3-panel visualizations from evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tasks = results.get('tasks', [])

        for i, task in enumerate(tasks):
            task_type = task.get('task_type', 'unknown')
            targets = task.get('expected_targets', {})
            score = task.get('score', 0)

            # Reconstruct input image with markers
            if 'allocentric' in task_type:
                vp = targets.get('viewpoint_position')
                facing = targets.get('facing', 'camera')
                if vp:
                    vp = (int(vp[0]), int(vp[1]))
                    annotator = ViewpointAnnotator(self.image_path)
                    annotator.add_viewpoint_marker(vp, facing)
                    input_img = annotator.get_image().convert('RGB')
                else:
                    input_img = self.base_image.copy()
            elif task_type == 'path_planning':
                start_pos = targets.get('start_position', (3000, 3400))
                start_pos = (int(start_pos[0]), int(start_pos[1]))
                annotator = ViewpointAnnotator(self.image_path)
                annotator.add_start_marker(start_pos, 'START HERE')
                input_img = annotator.get_image().convert('RGB')
            else:
                input_img = self.base_image.copy()

            # Create all 3 panels
            input_panel = self.create_input_panel(
                task_type=task_type,
                input_image=input_img,
                prompt=task.get('instruction', ''),
                instruction=task.get('instruction', '')
            )

            gt_panel = self.create_ground_truth_panel(
                task_type=task_type,
                base_image=self.base_image,
                expected_targets=targets
            )

            output_panel = self.create_output_panel(
                output_image=input_img,
                response=task.get('model_response', ''),
                outputs=task.get('recorded_outputs', {}),
                score=score,
                ground_truth=targets
            )

            # Combine 3 panels
            total_width = input_panel.width + gt_panel.width + output_panel.width + 30
            combined = Image.new('RGB', (total_width, input_panel.height), (15, 15, 15))
            combined.paste(input_panel, (0, 0))
            combined.paste(gt_panel, (input_panel.width + 15, 0))
            combined.paste(output_panel, (input_panel.width + gt_panel.width + 30, 0))

            # Save
            filename = f"task_{i+1:02d}_{task_type}_score{int(score*100)}.png"
            combined.save(output_path / filename)
            print(f"Created: {filename}")

        print(f"\nVisualization saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize spatial tasks (3-panel)")
    parser.add_argument("--sanity", action="store_true", help="Sanity check mode (INPUT + GROUND TRUTH)")
    parser.add_argument("--graph", "-g", help="Path to spatial_graph.json")
    parser.add_argument("--image", "-i", required=True, help="Path to scene image")
    parser.add_argument("--results", "-r", help="Path to evaluation results JSON")
    parser.add_argument("--output", "-o", default="task_visualizations", help="Output directory")
    parser.add_argument("--num-tasks", "-n", type=int, default=8, help="Number of tasks for sanity check")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    visualizer = TaskVisualizer(args.image, graph_path=args.graph)

    if args.sanity:
        if not args.graph:
            print("Error: --graph required for sanity check mode")
            return

        # Generate tasks for sanity check
        gt = GroundTruth(args.graph)
        generator = TaskGenerator(gt, image_path=args.image, seed=args.seed)

        # Generate 2 of each task type
        tasks = []
        for _ in range(max(1, args.num_tasks // 4)):
            tasks.append(generator.generate_localization_task())
            tasks.append(generator.generate_path_task())
            tasks.append(generator.generate_egocentric_qa_task())
            tasks.append(generator.generate_allocentric_qa_task())

        visualizer.visualize_sanity_check(tasks[:args.num_tasks], args.output)

    elif args.results:
        if not args.graph:
            print("Warning: --graph not provided, ground truth visualization may be limited")

        with open(args.results) as f:
            results = json.load(f)
        visualizer.visualize_results(results, args.output)

    else:
        print("Error: Specify --sanity for sanity check or --results for full visualization")


if __name__ == "__main__":
    main()
