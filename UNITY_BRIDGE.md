# Unity Bridge: Embodied Spatial Evaluation Pipeline

This document describes the Unity-based embodied evaluation system for spatial reasoning in Vision-Language Models.

## Overview

The Unity Bridge enables VLMs to reconstruct 3D spatial layouts by placing labeled spheres in a Unity scene. Given an image, the model uses tool calls to position objects, and we evaluate whether spatial relations (left/right, near/far, above/below) are preserved.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           UNITY EMBODIED PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    HTTP (localhost:5555)    ┌───────────────────────────┐ │
│  │   Python     │ ◄─────────────────────────► │   Unity Scene             │ │
│  │              │                             │                           │ │
│  │  UnityBridge │    place_object(...)        │  CherryUnityBridge.cs     │ │
│  │  (bridge.py) │    clear_scene()            │  CherryCamera.cs          │ │
│  │              │    get_scene_state()        │                           │ │
│  └──────┬───────┘    health()                 └───────────────────────────┘ │
│         │                                                                   │
│         │  LangChain Tools                                                  │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │   VLM Agent  │  (Claude / GPT-4o / Qwen)                                 │
│  │              │                                                           │
│  │  - Sees image                                                            │
│  │  - Calls tools to place objects                                          │
│  │  - Multi-turn agentic loop                                               │
│  └──────────────┘                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
unity_bridge/
├── __init__.py              # Exports: UnityBridge, PlacedObject, SceneState, create_unity_tools
├── bridge.py                # Python HTTP client for Unity
├── tools.py                 # LangChain-compatible tool definitions
├── CherryUnityBridge.cs     # Unity C# HTTP server (attach to GameObject)
└── CherryCamera.cs          # Unity camera controller (stationary, rotatable)
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
        ╱
       ╱
      ╱
   -Z (near/foreground)
```

| Axis | Range | Mapping |
|------|-------|---------|
| X | [-10, +10] | Left (-) ↔ Right (+) |
| Y | [0, +10] | Ground (0) → Up (+); sphere base at Y=0.5 |
| Z | [-10, +10] | Near/foreground (-) ↔ Far/background (+) |

### Image-to-Unity Mapping

| Image Position | Unity Coordinate |
|----------------|------------------|
| Left of frame | Negative X |
| Right of frame | Positive X |
| Bottom (foreground) | Negative Z |
| Top (background) | Positive Z |
| Physically higher | Positive Y |

## Components

### Python Side

#### `UnityBridge` (bridge.py)

Low-level HTTP client that communicates with Unity.

```python
from unity_bridge import UnityBridge

bridge = UnityBridge(base_url="http://localhost:5555")
bridge.wait_for_unity(timeout_s=30)      # Block until Unity ready
bridge.initialize_scene()                 # Clear and reset
bridge.place_object("chair", x=2, y=0.5, z=3, color="blue", scale=1.0)
bridge.place_object("table", x=-1, y=0.5, z=5, color="brown", scale=1.5)
state = bridge.get_scene_state()          # Returns SceneState
print(state.summary())
bridge.clear_scene()                      # Remove all objects
```

#### `create_unity_tools` (tools.py)

Factory function that creates LangChain-compatible tools for VLM agents.

```python
from unity_bridge import UnityBridge, create_unity_tools

bridge = UnityBridge()
bridge.wait_for_unity()
tools = create_unity_tools(bridge)  # Returns [place_object, clear_scene, get_scene_state]

# Pass to LangChain agent or provider.bind_tools()
```

**Available Tools:**

| Tool | Parameters | Description |
|------|------------|-------------|
| `place_object` | label, x, y, z, color?, scale? | Place a labeled sphere at (x,y,z) |
| `clear_scene` | - | Remove all placed spheres |
| `get_scene_state` | - | List all currently placed objects |

### Unity Side

#### `CherryUnityBridge.cs`

Attach this script to any GameObject in your Unity scene. It starts an HTTP server on port 5555.

**HTTP Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check, returns `{"status":"ok"}` |
| POST | `/place_object` | Place sphere: `{"label","x","y","z","color?","scale?"}` |
| POST | `/clear_scene` | Remove all spheres |
| GET | `/scene_state` | List all placed objects |
| POST | `/initialize` | Reset scene (alias for clear) |

#### `CherryCamera.cs`

Stationary camera controller attached automatically. Position is fixed at origin; left-click drag to rotate (yaw/pitch clamped to ±90°).

## Running the Pipeline

### Setup

1. **Unity Scene Setup:**
   - Create empty GameObject named "Bridge"
   - Attach `CherryUnityBridge.cs` to it
   - Press Play → HTTP server starts on port 5555

2. **Python Environment:**
   ```bash
   pip install langchain-core pydantic
   ```

### Basic Usage

```bash
# Single image with Claude (default)
python run_unity_eval.py --image photos/kitchen.jpg

# With spatial graph for better context
python run_unity_eval.py \
    --image photos/kitchen.jpg \
    --graph spatial_outputs/kitchen_spatial_graph.json

# Using GPT-4o
python run_unity_eval.py --image photos/living_room.jpg --provider openai

# Using Qwen via vLLM
python run_unity_eval.py --image photos/bedroom.jpg \
    --provider vllm --model Qwen/Qwen2.5-VL-7B-Instruct
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--image, -i` | (required) | Path to scene image |
| `--graph, -g` | None | Path to spatial_graph.json |
| `--provider, -p` | claude | VLM: claude, openai, qwen, vllm, ollama |
| `--model, -m` | (provider default) | Specific model name |
| `--unity-port` | 5555 | Unity bridge port |
| `--unity-url` | None | Full Unity URL (overrides port) |
| `--max-turns, -n` | 10 | Max agentic turns |
| `--output-dir, -o` | unity_eval_outputs | Output directory |
| `--quiet, -q` | False | Suppress verbose output |

## Evaluation Metrics

The pipeline compares Unity placements against spatial graph ground truth:

| Metric | Description |
|--------|-------------|
| `coverage` | Fraction of graph entities that have a placed object |
| `left_right_accuracy` | Fraction of pairwise left/right relations preserved |
| `near_far_accuracy` | Fraction of pairwise near/far (depth) relations preserved |

## Output Format

Results are saved to `unity_eval_outputs/unity_eval_{image_stem}.json`:

```json
{
  "image": "photos/kitchen.jpg",
  "graph": "spatial_outputs/kitchen_spatial_graph.json",
  "provider": "claude",
  "model": "claude-3-5-sonnet-20241022",
  "placed_objects": [
    {"id": 0, "label": "refrigerator", "x": -3.0, "y": 0.5, "z": 5.0, "color": "#E63333", "scale": 2.0},
    {"id": 1, "label": "table", "x": 1.0, "y": 0.5, "z": 3.0, "color": "#3380E6", "scale": 1.5}
  ],
  "evaluation": {
    "objects_placed": 5,
    "entities_in_graph": 8,
    "coverage": 0.625,
    "left_right_accuracy": 0.8,
    "near_far_accuracy": 0.7,
    "pairwise_pairs_evaluated": 10
  }
}
```

## Integration with Spatial Query Benchmark

Future integration points:

1. **Input**: Feed `query_benchmark/` generated queries as evaluation targets
2. **Grounding**: Use Unity 3D coordinates as ground truth for spatial relations
3. **Embodied QA**: Add tool `answer_spatial_query(relation)` for the model to respond to queries based on placed objects
4. **Allocentric Evaluation**: Test perspective-taking by querying relations from different anchor viewpoints

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Port 5555 in use" | Stop Unity Play mode, wait 5 seconds, restart |
| "Cannot reach Unity" | Ensure Unity scene is in Play mode with CherryUnityBridge attached |
| Pink/magenta spheres | Shader compatibility issue; script auto-detects URP/HDRP/Standard |
| Timeout errors | Increase `--unity-timeout` or check Unity console for errors |
