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
  Y :  Ground (0) … Up (+10)   — sphere sits on ground at Y = 0.5
  Z :  Near  (0)  … Far (+20)  — Z=0 at camera, Z=20 is far background

Setup
-----
1. Open Unity, load a scene that has CherryUnityBridge.cs attached to any
   GameObject (e.g. an empty "Bridge" object).
2. Press Play.  The script starts an HTTP server on http://localhost:5555/.
3. Run this script from the cherry/ directory.

Examples
--------
    # Zero-shot (default), single image
    python run_unity_eval.py --image photos/kitchen.jpg

    # Multi-turn refinement (3 turns)
    python run_unity_eval.py --image photos/kitchen.jpg --mode multi-turn --max-turns 3

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
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

# Load .env if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from unity_bridge import UnityBridge, create_unity_tools, SceneState
from spatial_agent.model_providers import create_model_provider, VLMProvider
from spatial_agent.prompts import ZERO_SHOT_PLACEMENT_SYSTEM_PROMPT, MULTI_TURN_PLACEMENT_SYSTEM_PROMPT
from spatial_graph_to_unity import convert_for_evaluation


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clamp_coords(args: Dict) -> Dict:
    """Clamp x/y/z to valid Unity ranges in-place."""
    args["x"] = max(-10.0, min(10.0, float(args.get("x", 0))))
    args["y"] = max(0.0,   min(10.0, float(args.get("y", 0.5))))
    args["z"] = max(0.0,   min(20.0, float(args.get("z", 5))))
    return args


def _tools_to_schema(tools) -> List[Dict]:
    """Convert LangChain BaseTool list to OpenAI-compatible function schema dicts."""
    schemas = []
    for tool in tools:
        schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
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


def _dispatch_tool(name: str, args: Dict, tools) -> str:
    """Find and call the matching tool."""
    for tool in tools:
        if tool.name == name:
            try:
                return tool._run(**args)
            except TypeError as exc:
                return f"Bad arguments for {name}: {exc}"
    return f"Unknown tool: {name}"


def take_snapshot(bridge: UnityBridge, out_dir: Path, turn: int) -> str:
    """Capture a Unity scene screenshot and save it to out_dir."""
    png_bytes = bridge.capture_view()
    path = out_dir / f"snapshot_turn_{turn:02d}.png"
    path.write_bytes(png_bytes)
    return str(path)


def composite_images(ref_path: str, snap_path: str, out_path: str) -> str:
    """
    Place ref image (left) and snapshot (right) side-by-side into a single PNG.
    Falls back to just the snapshot if PIL is unavailable.
    """
    try:
        from PIL import Image
        ref = Image.open(ref_path).convert("RGB")
        snap = Image.open(snap_path).convert("RGB")
        h = max(ref.height, snap.height)
        ref = ref.resize((int(ref.width * h / ref.height), h))
        snap = snap.resize((int(snap.width * h / snap.height), h))
        combined = Image.new("RGB", (ref.width + snap.width, h))
        combined.paste(ref, (0, 0))
        combined.paste(snap, (ref.width, 0))
        combined.save(out_path)
        return out_path
    except ImportError:
        return snap_path


# ─── Zero-shot placement ──────────────────────────────────────────────────────

def run_zero_shot(
    image_path: str,
    provider: VLMProvider,
    bridge: UnityBridge,
    graph_data: Optional[Dict] = None,
    verbose: bool = True,
) -> SceneState:
    """
    Zero-shot placement: one API call, execute all returned place_object calls.

    Returns the final SceneState.
    """
    tools = create_unity_tools(bridge)
    tool_schemas = _tools_to_schema(tools)

    user_prompt = "Reconstruct the spatial layout of the scene by placing objects in Unity."

    if graph_data:
        entities = graph_data.get("entities", [])
        labels = [e.get("label", "") for e in entities[:12] if e.get("label")]
        if labels:
            user_prompt += (
                f"\n\nDetected objects: {', '.join(labels)}. "
                "Place these objects with accurate relative positions."
            )

    if verbose:
        print(f"\n[Unity Eval] Image:    {image_path}")
        print(f"[Unity Eval] Provider: {provider.model_name}")

    response = provider.generate(
        image_path=image_path,
        prompt=user_prompt,
        system_prompt=ZERO_SHOT_PLACEMENT_SYSTEM_PROMPT,
        tools=tool_schemas,
    )

    if verbose and response.text:
        print(f"[Model] {response.text}")

    for tc in response.tool_calls:
        if tc.name != "place_object":
            continue
        args = _clamp_coords(dict(tc.arguments))
        if verbose:
            print(f"  → place_object({args})")
        result = _dispatch_tool(tc.name, args, tools)
        if verbose:
            print(f"     {result}")

    final_state = bridge.get_scene_state()
    if verbose:
        print(f"\n[Unity Eval] Placed {final_state.count} object(s).")

    return final_state


# ─── Multi-turn placement ─────────────────────────────────────────────────────

def run_multi_turn(
    image_path: str,
    provider: VLMProvider,
    bridge: UnityBridge,
    out_dir: Path,
    graph_data: Optional[Dict] = None,
    max_turns: int = 3,
    verbose: bool = True,
) -> Tuple[SceneState, List[str]]:
    """
    Multi-turn placement: zero-shot first, then iterative snapshot + refine.

    Returns (final_scene_state, list_of_snapshot_paths).
    """
    # Step 1: zero-shot initial placement
    run_zero_shot(image_path, provider, bridge, graph_data=graph_data, verbose=verbose)

    snapshots: List[str] = []
    tools = create_unity_tools(bridge)
    tool_schemas = _tools_to_schema(tools)

    for turn in range(max_turns):
        snap_path = take_snapshot(bridge, out_dir, turn)
        snapshots.append(snap_path)
        if verbose:
            print(f"\n[Turn {turn + 1}] Snapshot: {snap_path}")

        # Composite reference + snapshot into one image for single-image APIs
        composite_path = str(out_dir / f"composite_turn_{turn:02d}.png")
        composite_images(image_path, snap_path, composite_path)

        # Build user prompt with current scene state so model knows coordinates
        scene = bridge.get_scene_state()
        scene_lines = [
            f"  {o.label}: x={o.x:.1f}, y={o.y:.1f}, z={o.z:.1f}"
            for o in scene.objects
        ]
        user_prompt = "Current objects:\n" + "\n".join(scene_lines) if scene_lines else "Current objects: (none)"

        response = provider.generate(
            image_path=composite_path,
            prompt=user_prompt,
            system_prompt=MULTI_TURN_PLACEMENT_SYSTEM_PROMPT,
            tools=tool_schemas,
        )

        if verbose and response.text:
            print(f"[Model] {response.text}")

        if not response.tool_calls:
            if verbose:
                print("[Multi-turn] No tool calls — stopping early.")
            break

        for tc in response.tool_calls:
            args = _clamp_coords(dict(tc.arguments))
            if verbose:
                print(f"  → {tc.name}({args})")
            result = _dispatch_tool(tc.name, args, tools)
            if verbose:
                print(f"     {result}")

    # Final snapshot
    final_snap = take_snapshot(bridge, out_dir, len(snapshots))
    snapshots.append(final_snap)

    return bridge.get_scene_state(), snapshots


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
                gt_lr = ea["x"] < eb["x"]
                pred_lr = oa.x < ob.x
                lr_correct += int(gt_lr == pred_lr)
                lr_total += 1

            # Near-far (Z axis ↔ depth / z_order in graph)
            if ea.get("z_order") is not None and eb.get("z_order") is not None:
                gt_nf = ea["z_order"] < eb["z_order"]
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
                        choices=["claude", "openai", "huggingface", "hf", "vllm", "qwen", "ollama"],
                        default="huggingface",
                        help="VLM provider (default: huggingface)")
    parser.add_argument("--model", "-m",
                        help="Model name (uses provider default if omitted)")
    parser.add_argument("--hf-provider", default="auto",
                        help="HuggingFace inference router (e.g. fireworks-ai, together, hf-inference). Default: auto")
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

    # Pipeline mode
    parser.add_argument("--mode", choices=["zero-shot", "multi-turn"], default="zero-shot",
                        help="Placement mode (default: zero-shot)")
    parser.add_argument("--max-turns", type=int, default=3,
                        help="Number of refinement turns for multi-turn mode (default: 3)")

    # Eval
    parser.add_argument("--output-dir", "-o", default="unity_eval_outputs",
                        help="Base directory for output (default: unity_eval_outputs)")
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
            raw_graph = json.load(f)
        # Normalise: depth_sam3_connector.py uses "nodes"; evaluation expects "entities"
        if "nodes" in raw_graph and "entities" not in raw_graph:
            graph_data = convert_for_evaluation(raw_graph)
        else:
            graph_data = raw_graph

    # ── Per-image output directory ────────────────────────────────────────────
    out_dir = Path(args.output_dir) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

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
    if args.provider in ("huggingface", "hf"):
        provider_kwargs["hf_provider"] = args.hf_provider

    provider = create_model_provider(
        args.provider,
        model_name=args.model,
        **provider_kwargs,
    )

    # ── Run placement ────────────────────────────────────────────────────────
    snapshots: List[str] = []

    if args.mode == "multi-turn":
        final_state, snapshots = run_multi_turn(
            image_path=str(image_path),
            provider=provider,
            bridge=bridge,
            out_dir=out_dir,
            graph_data=graph_data,
            max_turns=args.max_turns,
            verbose=not args.quiet,
        )
    else:
        final_state = run_zero_shot(
            image_path=str(image_path),
            provider=provider,
            bridge=bridge,
            graph_data=graph_data,
            verbose=not args.quiet,
        )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    eval_results = evaluate_placement(final_state, graph_data)
    if not args.quiet:
        print("\n[Unity Eval] Evaluation results:")
        for k, v in eval_results.items():
            print(f"  {k}: {v}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = out_dir / f"unity_eval_{image_path.stem}.json"

    output = {
        "image": str(image_path),
        "mode": args.mode,
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

    if snapshots:
        output["snapshots"] = snapshots

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[Unity Eval] Results saved to: {out_path}")


if __name__ == "__main__":
    main()
