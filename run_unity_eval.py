#!/usr/bin/env python3
"""
run_unity_eval.py - Embodied evaluation via Unity 3D scene placement

Given an image (+ optional spatial_graph.json), a VLM is prompted to place
labelled spheres in a Unity scene to reconstruct the 3D spatial layout of the
objects visible in the image.

The model uses three tools:
  place_object(label, x, y, z, color?, scale?) — place a sphere
  clear_scene()                                 — wipe all spheres
  get_scene_state()                             — inspect current placements

Unity Coordinate System
-----------------------
  X :  Left (−10) … Right (+10)
  Y :  Ground (0) … Up (+10)    — sphere sits on ground at Y = 0.5
  Z :  Near  (−10) … Far (+10)  — depth / foreground–background

Setup
-----
1. Open Unity, load a scene that has CherryUnityBridge.cs attached to any
   GameObject (e.g. an empty "Bridge" object).
2. Press Play.  The script starts an HTTP server on http://localhost:5555/.
3. Run this script from the cherry/ directory.

Examples
--------
    # Claude (default), single image
    python run_unity_eval.py --image photos/kitchen.jpg

    # Claude with a spatial graph for richer context
    python run_unity_eval.py \\
        --image photos/kitchen.jpg \\
        --graph spatial_outputs/kitchen_spatial_graph.json

    # GPT-4o
    python run_unity_eval.py --image photos/living_room2.jpg --provider openai

    # Qwen via vLLM server
    python run_unity_eval.py --image photos/bathroom_2.jpg \\
        --provider vllm --model Qwen/Qwen3-VL-8B-Instruct

    # Custom Unity port
    python run_unity_eval.py --image photos/kitchen.jpg --unity-port 5556
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict

sys.path.insert(0, str(Path(__file__).parent))

from unity_bridge import UnityBridge, create_unity_tools, SceneState
from spatial_agent.model_providers import create_model_provider, VLMProvider


# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an embodied spatial reasoning agent.

Your task is to reconstruct the 3D spatial layout of a scene by placing
labelled spheres in a Unity landscape.  You will be given an image (and
optionally a text description) of a real scene.

## Your job
Identify the key objects in the image and place each one as a labelled sphere
whose position encodes its spatial relationship to the others.

## Unity coordinate system
The scene is a flat 20 × 20 unit ground plane.  Coordinate axes:

  X axis  :  Left (−10)  ←→  Right (+10)
  Y axis  :  Ground (0)  ↑   Up (+10)
  Z axis  :  Near  (−10) ←→  Far  (+10)   [depth away from camera]

Guidelines for mapping image space → Unity space:
  • Objects on the LEFT  of the image → more negative X
  • Objects on the RIGHT of the image → more positive X
  • Objects in the FOREGROUND (bottom of image) → more negative Z
  • Objects in the BACKGROUND (top of image)    → more positive Z
  • Objects that are physically HIGHER → more positive Y
  • Spheres sit on the ground at Y = 0.5 (sphere radius).
  • Use scale to reflect object size (table → 1.5–2.0, cup → 0.5).

## Strategy
1. Scan the image for distinct objects worth representing.
2. Plan their relative positions mentally (what's to the left of what? what's
   closer to the camera? what's stacked on top of what?).
3. Call place_object for each object, choosing (x, y, z) to encode those
   relationships accurately.
4. Call get_scene_state to review your placements.
5. Optionally adjust with clear_scene + re-place if needed.
6. When satisfied, state "DONE" in your final message.

## Important
- Place at least the 5 most spatially prominent objects.
- Keep positions within scene bounds: X ∈ [−10,10], Z ∈ [−10,10], Y ∈ [0,10].
- Objects should NOT overlap (keep at least 1 unit of separation).
- Labels should be concise (e.g. "chair", "table", "window", "refrigerator").
"""


# ─── Agent loop ───────────────────────────────────────────────────────────────

def run_agentic_placement(
    image_path: str,
    provider: VLMProvider,
    bridge: UnityBridge,
    graph_data: Optional[Dict] = None,
    max_turns: int = 10,
    verbose: bool = True,
) -> SceneState:
    """
    Run a multi-turn agentic loop where the VLM places objects in Unity.

    The loop continues until the model outputs "DONE" or max_turns is reached.
    Tool calls are executed against the live Unity scene.

    Returns the final SceneState.
    """
    tools = create_unity_tools(bridge)

    # Build the user prompt
    user_prompt = "Please reconstruct the spatial layout of the scene shown in the image by placing labelled spheres in Unity."

    if graph_data:
        entities = graph_data.get("entities", [])
        if entities:
            labels = [e.get("label", "") for e in entities[:12] if e.get("label")]
            if labels:
                user_prompt += (
                    f"\n\nThe following objects have been detected in the scene "
                    f"(from the spatial graph): {', '.join(labels)}. "
                    "Focus on placing these objects with accurate relative positions."
                )

    if verbose:
        print(f"\n[Unity Eval] Image: {image_path}")
        print(f"[Unity Eval] Provider: {provider.model_name}")
        print(f"[Unity Eval] Max turns: {max_turns}")
        print("[Unity Eval] Starting placement loop…\n")

    # Convert LangChain tools to provider tool schema
    tool_schemas = _tools_to_schema(tools)

    # Conversation history (for multi-turn providers)
    conversation: List[Dict] = []

    for turn in range(1, max_turns + 1):
        if verbose:
            print(f"── Turn {turn}/{max_turns} ────────────────────────────────────")

        # On first turn, send image + prompt; subsequent turns only send tool results
        current_prompt = user_prompt if turn == 1 else _build_followup_prompt(conversation)

        response = provider.generate(
            image_path=image_path,
            prompt=current_prompt,
            system_prompt=SYSTEM_PROMPT,
            tools=tool_schemas,
        )

        if verbose and response.text:
            print(f"[Model] {response.text}")

        # Execute tool calls
        any_tool_called = False
        for tc in response.tool_calls:
            any_tool_called = True
            if verbose:
                print(f"  → Tool: {tc.name}({tc.arguments})")

            result = _dispatch_tool(tc.name, tc.arguments, tools)

            if verbose:
                print(f"     ✓ {result}")

            conversation.append({
                "tool": tc.name,
                "args": tc.arguments,
                "result": result,
            })

        # Check termination
        text_lower = response.text.lower() if response.text else ""
        if "done" in text_lower and not any_tool_called:
            if verbose:
                print("\n[Unity Eval] Model signalled DONE.")
            break

        if not any_tool_called and not response.text:
            if verbose:
                print("[Unity Eval] No tool calls and no text — stopping.")
            break

    # Final state
    final_state = bridge.get_scene_state()
    if verbose:
        print(f"\n[Unity Eval] Final scene state:\n{final_state.summary()}")

    return final_state


def _tools_to_schema(tools) -> List[Dict]:
    """Convert LangChain BaseTool list to OpenAI-compatible function schema dicts."""
    schemas = []
    for tool in tools:
        schema = tool.args_schema.schema() if tool.args_schema else {}
        schemas.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        })
    return schemas


def _build_followup_prompt(conversation: List[Dict]) -> str:
    """Build a follow-up prompt from the tool call history."""
    if not conversation:
        return "Please continue placing objects."
    last = conversation[-1]
    return (
        f"Tool '{last['tool']}' returned: {last['result']}\n\n"
        "Continue placing objects or call get_scene_state to review, "
        "then say DONE when finished."
    )


def _dispatch_tool(name: str, args: Dict, tools) -> str:
    """Find and call the matching tool."""
    for tool in tools:
        if tool.name == name:
            try:
                return tool._run(**args)
            except TypeError as exc:
                return f"Bad arguments for {name}: {exc}"
    return f"Unknown tool: {name}"


# ─── Evaluation: compare Unity placement to spatial graph ─────────────────────

def evaluate_placement(final_state: SceneState, graph_data: Optional[Dict]) -> Dict:
    """
    Compare placed object positions against the spatial graph ground truth.

    Metrics:
      coverage     : fraction of graph entities that have a matching placed object
      left_right   : fraction of left/right relations preserved
      near_far     : fraction of near/far (foreground/background) relations preserved
      up_down      : fraction of above/below relations preserved (Y axis)
    """
    if not graph_data or not final_state.objects:
        return {"coverage": 0.0, "left_right": None, "near_far": None, "up_down": None}

    entities = graph_data.get("entities", [])
    placed_labels = [o.label.lower() for o in final_state.objects]

    # Coverage: how many entities were placed?
    matched = sum(
        1 for e in entities
        if any(e.get("label", "").lower() in pl or pl in e.get("label", "").lower()
               for pl in placed_labels)
    )
    coverage = matched / len(entities) if entities else 0.0

    # Pairwise spatial relation checks
    lr_correct, lr_total = 0, 0
    nf_correct, nf_total = 0, 0
    ud_correct, ud_total = 0, 0

    placed_map = {o.label.lower(): o for o in final_state.objects}

    for i, ea in enumerate(entities):
        for eb in entities[i + 1:]:
            la = ea.get("label", "").lower()
            lb = eb.get("label", "").lower()
            oa = _find_placed(la, placed_map)
            ob = _find_placed(lb, placed_map)
            if oa is None or ob is None:
                continue

            # Left-right (X axis)
            if ea.get("x") is not None and eb.get("x") is not None:
                gt_lr = ea["x"] < eb["x"]           # ea is to the left
                pred_lr = oa.x < ob.x
                lr_correct += int(gt_lr == pred_lr)
                lr_total += 1

            # Near-far (Z axis ↔ depth / z_order in graph)
            if ea.get("z_order") is not None and eb.get("z_order") is not None:
                gt_nf = ea["z_order"] < eb["z_order"]   # ea is closer
                pred_nf = oa.z < ob.z
                nf_correct += int(gt_nf == pred_nf)
                nf_total += 1

    results = {
        "objects_placed": final_state.count,
        "entities_in_graph": len(entities),
        "coverage": round(coverage, 3),
        "left_right_accuracy": round(lr_correct / lr_total, 3) if lr_total else None,
        "near_far_accuracy": round(nf_correct / nf_total, 3) if nf_total else None,
        "pairwise_pairs_evaluated": lr_total + nf_total,
    }
    return results


def _find_placed(label: str, placed_map: Dict):
    """Fuzzy match: exact, then substring."""
    if label in placed_map:
        return placed_map[label]
    for k, v in placed_map.items():
        if label in k or k in label:
            return v
    return None


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cherry Unity Embodied Evaluation — place objects in Unity 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    parser.add_argument("--image", "-i", required=True,
                        help="Path to scene image")
    parser.add_argument("--graph", "-g",
                        help="Path to spatial_graph.json (optional but improves prompting)")

    # Model
    parser.add_argument("--provider", "-p",
                        choices=["claude", "qwen", "vllm", "openai", "ollama"],
                        default="claude",
                        help="VLM provider (default: claude)")
    parser.add_argument("--model", "-m",
                        help="Model name (uses provider default if omitted)")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1",
                        help="vLLM server URL (default: http://localhost:8000/v1)")

    # Unity
    parser.add_argument("--unity-port", type=int, default=5555,
                        help="Port where CherryUnityBridge is listening (default: 5555)")
    parser.add_argument("--unity-url",
                        help="Full Unity bridge URL (overrides --unity-port)")
    parser.add_argument("--unity-timeout", type=float, default=30.0,
                        help="Seconds to wait for Unity to become ready (default: 30)")
    parser.add_argument("--no-init", action="store_true",
                        help="Skip re-initializing the Unity scene (use current state)")

    # Eval
    parser.add_argument("--max-turns", "-n", type=int, default=10,
                        help="Max agentic turns (default: 10)")
    parser.add_argument("--output-dir", "-o", default="unity_eval_outputs",
                        help="Directory for output JSON (default: unity_eval_outputs)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    # ── Validate inputs ──────────────────────────────────────────────────────
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    graph_data = None
    if args.graph:
        graph_path = Path(args.graph)
        if not graph_path.exists():
            print(f"ERROR: graph not found: {graph_path}", file=sys.stderr)
            sys.exit(1)
        with open(graph_path) as f:
            graph_data = json.load(f)

    # ── Unity bridge ─────────────────────────────────────────────────────────
    unity_url = args.unity_url or f"http://localhost:{args.unity_port}"
    bridge = UnityBridge(base_url=unity_url)
    print(f"[Unity Eval] Connecting to Unity at {unity_url}…")
    bridge.wait_for_unity(timeout_s=args.unity_timeout)

    if not args.no_init:
        bridge.initialize_scene()

    # ── Model provider ───────────────────────────────────────────────────────
    provider_kwargs = {}
    if args.provider == "vllm":
        provider_kwargs["base_url"] = args.vllm_url

    provider = create_model_provider(
        args.provider,
        model_name=args.model,
        **provider_kwargs,
    )

    # ── Run placement ────────────────────────────────────────────────────────
    final_state = run_agentic_placement(
        image_path=str(image_path),
        provider=provider,
        bridge=bridge,
        graph_data=graph_data,
        max_turns=args.max_turns,
        verbose=not args.quiet,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    eval_results = evaluate_placement(final_state, graph_data)
    if not args.quiet:
        print("\n[Unity Eval] Evaluation results:")
        for k, v in eval_results.items():
            print(f"  {k}: {v}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"unity_eval_{image_path.stem}.json"

    output = {
        "image": str(image_path),
        "graph": args.graph,
        "provider": args.provider,
        "model": args.model or "default",
        "placed_objects": [
            {"id": o.id, "label": o.label,
             "x": o.x, "y": o.y, "z": o.z,
             "color": o.color, "scale": o.scale}
            for o in final_state.objects
        ],
        "evaluation": eval_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[Unity Eval] Results saved to: {out_path}")


if __name__ == "__main__":
    main()
