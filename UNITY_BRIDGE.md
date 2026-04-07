# Unity Bridge: Embodied Spatial Evaluation Pipeline

This document describes the Unity-based embodied evaluation system for spatial reasoning in Vision-Language Models.

## Overview

The Unity Bridge enables VLMs to reconstruct 3D spatial layouts by placing labeled spheres in a Unity scene. Two evaluation modes are supported:

- **Zero-shot**: one API call → model places all objects
- **Multi-turn**: zero-shot → snapshot → model refines → snapshot → repeat

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           UNITY EMBODIED PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    HTTP (localhost:5555)    ┌───────────────────────────┐ │
│  │   Python     │ ◄─────────────────────────► │   Unity Scene             │ │
│  │              │                             │                           │ │
│  │  UnityBridge │    place_object(...)        │  CherryUnityBridge.cs     │ │
│  │  (bridge.py) │    remove_object(...)       │                           │ │
│  │              │    move_object(...)         │                           │ │
│  │              │    capture_view()           │                           │ │
│  └──────┬───────┘                             └───────────────────────────┘ │
│         │                                                                   │
│         │  LangChain Tools                                                  │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │   VLM Agent  │  (Claude / GPT-4o / Qwen via HuggingFace or vLLM)        │
│  │              │                                                           │
│  │  Zero-shot:  places objects from image in one pass                       │
│  │  Multi-turn: sees snapshot each turn, refines placements iteratively     │
│  └──────────────┘                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Modes

### Zero-shot

```
Image ──► VLM ──► place_object × N ──► evaluate
```

The model receives the scene image once and emits all `place_object` tool calls in a single response.

### Multi-turn

```
Image ──► VLM ──► place_object × N   (zero-shot seed)
                       │
              ┌────────▼────────┐
              │  capture_view() │  → snapshot_turn_00.png
              └────────┬────────┘
                       │  [model sees snapshot, refines]
              ┌────────▼────────────────────────────┐
              │  move_object / remove_object /       │
              │  place_object × M                   │
              └────────┬────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  capture_view() │  → snapshot_turn_01.png
              └────────┬────────┘
                       │  ...repeat up to --max-turns
              ┌────────▼────────┐
              │  capture_view() │  → snapshot_turn_final.png
              └─────────────────┘
```

Each refinement turn receives **only the current Unity snapshot** (no text prompt, no coordinate list). The model's full conversation history is maintained across turns so it can reason about what it has already done.

## Directory Structure

```
unity_bridge/
├── __init__.py              # Exports: UnityBridge, PlacedObject, SceneState, create_unity_tools
├── bridge.py                # Python HTTP client for Unity
├── tools.py                 # LangChain-compatible tool definitions
└── CherryUnityBridge.cs     # Unity C# HTTP server (attach to GameObject)

spatial_agent/
├── __init__.py
├── model_providers.py       # VLMProvider base + 5 provider implementations
└── prompts.py               # ZERO_SHOT_PLACEMENT_SYSTEM_PROMPT, MULTI_TURN_PLACEMENT_SYSTEM_PROMPT

run_unity_eval.py            # CLI entry point
```

## Coordinate System

Unity uses a left-handed coordinate system with the camera at the origin:

```
        +Y (up)
         │
         │        +Z (far/background)
         │       ╱
         │      ╱
         │     ╱
         └───────────── +X (right)
```

| Axis | Range | Mapping |
|------|-------|---------|
| X | [-10, +10] | Left (-) ↔ Right (+) |
| Y | [0, +10] | Ground (0) → Up (+); object base at Y=0.5 |
| Z | [0, +20] | Near camera (0) → Far background (+20) |

## Components

### Python Side

#### `UnityBridge` (bridge.py)

Low-level HTTP client that communicates with Unity.

```python
from unity_bridge import UnityBridge

bridge = UnityBridge(base_url="http://localhost:5555")
bridge.wait_for_unity(timeout_s=30)      # Block until Unity ready
bridge.initialize_scene()                 # Clear and reset

bridge.place_object("chair", x=2, y=0.5, z=3, scale=1.0)
bridge.remove_object(x=2, y=0.5, z=3)
bridge.move_object(x=2, y=0.5, z=3, new_x=4, new_y=0.5, new_z=5)

state = bridge.get_scene_state()          # Returns SceneState
png_bytes = bridge.capture_view()         # Returns PNG bytes
bridge.clear_scene()
```

#### `create_unity_tools` (tools.py)

Returns three LangChain-compatible tools used by the VLM agent:

| Tool | Parameters | Description |
|------|------------|-------------|
| `place_object` | label, x, y, z, scale?, shape? | Place a labeled sphere/cube at (x,y,z) |
| `remove_object` | x, y, z | Remove the object closest to (x,y,z) |
| `move_object` | x, y, z, new_x, new_y, new_z | Move the object closest to (x,y,z) |

Zero-shot uses only `place_object`. Multi-turn has access to all three.

#### `model_providers.py`

Unified VLM provider interface. All providers share the same signature:

```python
provider.generate(
    image_path="photos/kitchen.jpg",   # path to scene image
    prompt="...",                       # user prompt
    system_prompt="...",               # system instruction
    tools=[...],                        # tool schemas
    snapshot_path=None,                # optional: inject second image
    messages=None,                     # optional: full conversation history (overrides above)
) -> VLMResponse
```

`VLMResponse` contains:
- `text` — model's text output
- `tool_calls` — list of `ToolCall(name, arguments, id)`
- `raw_assistant_message` — OpenAI-format dict for appending to conversation history

Supported providers:

| Alias | Class | Default Model |
|-------|-------|---------------|
| `claude` | `ClaudeProvider` | claude-sonnet-4-6 |
| `openai` | `OpenAIProvider` | gpt-4o |
| `huggingface` / `hf` | `HuggingFaceProvider` | Qwen/Qwen3-VL-8B-Instruct |
| `vllm` / `qwen` | `VLLMProvider` | Qwen/Qwen2-VL-7B-Instruct |
| `ollama` | `OllamaProvider` | qwen2-vl:7b |

### Unity Side

#### `CherryUnityBridge.cs`

Attach this script to any GameObject. It starts an HTTP server on port 5555.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/place_object` | Place object: `{"label","x","y","z","scale?","shape?"}` |
| POST | `/remove_object` | Remove nearest object: `{"x","y","z"}` |
| POST | `/move_object` | Move nearest object: `{"x","y","z","new_x","new_y","new_z"}` |
| POST | `/clear_scene` | Remove all objects |
| GET | `/scene_state` | List all placed objects |
| POST | `/initialize` | Reset scene |
| GET | `/capture_view` | Capture camera as base64 PNG |

## Running the Pipeline

### Setup

1. Open Unity, attach `CherryUnityBridge.cs` to a GameObject, press Play.
2. Install Python deps: `pip install langchain-core pydantic openai`

### Usage

```bash
# Zero-shot (default)
python run_unity_eval.py --image photos/kitchen.jpg --provider huggingface

# Multi-turn, 3 refinement turns
python run_unity_eval.py --image photos/kitchen.jpg \
    --provider huggingface --mode multi-turn --max-turns 3

# With spatial graph ground truth
python run_unity_eval.py \
    --image photos/kitchen.jpg \
    --graph spatial_outputs/kitchen_spatial_graph.json

# GPT-4o
python run_unity_eval.py --image photos/living_room.jpg --provider openai

# Qwen via local vLLM server
python run_unity_eval.py --image photos/bedroom.jpg \
    --provider vllm --model Qwen/Qwen2.5-VL-7B-Instruct
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--image, -i` | (required) | Path to scene image |
| `--graph, -g` | None | Path to spatial_graph.json |
| `--provider, -p` | `huggingface` | VLM provider: claude, openai, huggingface, vllm, ollama |
| `--model, -m` | (provider default) | Specific model name |
| `--mode` | `zero-shot` | Pipeline mode: `zero-shot` or `multi-turn` |
| `--max-turns` | `3` | Refinement turns for multi-turn mode |
| `--unity-port` | `5555` | Unity bridge port |
| `--unity-url` | None | Full Unity URL (overrides `--unity-port`) |
| `--unity-timeout` | `30` | Seconds to wait for Unity |
| `--no-init` | False | Skip scene reset |
| `--output-dir, -o` | `unity_eval_outputs` | Base output directory |
| `--quiet, -q` | False | Suppress verbose output |

## Output Format

Results are saved to `unity_eval_outputs/<image_stem>/unity_eval_<image_stem>.json`:

```json
{
  "image": "photos/kitchen.jpg",
  "mode": "multi-turn",
  "graph": null,
  "provider": "huggingface",
  "model": "default",
  "placed_objects": [
    {"id": 0, "label": "person", "x": 1.5, "y": 1.0, "z": 6.0, "color": "#E63333", "scale": 1.0},
    {"id": 1, "label": "kitchen_counter", "x": -0.5, "y": 0.5, "z": 3.5, "color": "#3380E6", "scale": 1.0}
  ],
  "evaluation": {
    "objects_placed": 8,
    "entities_in_graph": 0,
    "coverage": 0.0,
    "left_right_accuracy": null,
    "near_far_accuracy": null,
    "pairwise_pairs_evaluated": 0
  },
  "snapshots": [
    "unity_eval_outputs/kitchen/snapshot_turn_00.png",
    "unity_eval_outputs/kitchen/snapshot_turn_01.png",
    "unity_eval_outputs/kitchen/snapshot_turn_02.png",
    "unity_eval_outputs/kitchen/snapshot_turn_03.png"
  ]
}
```

The `snapshots` key is only present in multi-turn mode. Snapshots are taken after each refinement turn plus one final snapshot at the end.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `coverage` | Fraction of ground-truth entities that have a placed object (requires `--graph`) |
| `left_right_accuracy` | Fraction of pairwise left/right relations preserved |
| `near_far_accuracy` | Fraction of pairwise near/far (depth) relations preserved |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Port 5555 in use" | Stop Unity Play mode, wait a few seconds, restart |
| "Cannot reach Unity" | Ensure Unity scene is in Play mode with CherryUnityBridge.cs attached |
| Empty model response | Transient HuggingFace API issue — retry the run |
| Pink/magenta spheres | Shader compatibility issue; script auto-detects URP/HDRP/Standard |
| Timeout errors | Increase `--unity-timeout` or check Unity console for errors |
