# Cherry: VLM Spatial Understanding Evaluation Plan

This document outlines the research plan for evaluating Vision-Language Models' 3D spatial understanding through embodied interaction in Unity.

## Research Goal

**Test whether VLMs can accurately infer 3D spatial relationships from 2D images by having them externalize their understanding in a 3D environment.**

The key insight: Rather than asking VLMs to answer spatial questions textually (which may just test language priors), we have them "act out" their understanding by placing objects in a 3D scene—like a human reconstructing a room layout from memory.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     VLM SPATIAL UNDERSTANDING EVALUATION                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   INPUT                          VLM AS EMBODIED AGENT                          │
│   ─────                          ────────────────────                           │
│                                                                                  │
│   ┌─────────────┐                ┌────────────────────────────────────────┐     │
│   │   Image     │                │  VLM receives:                         │     │
│   │   (scene)   │───────────────►│    - Image of scene                    │     │
│   └─────────────┘                │    - System prompt                     │     │
│                                  │    - Unity tools (place/move/remove)   │     │
│                                  │                                        │     │
│                                  │  VLM generates:                        │     │
│                                  │    - Scene program (tool calls)        │     │
│                                  │    + Refinements from snapshots        │     │
│                                  └───────────────┬────────────────────────┘     │
│                                                  │                              │
│                                                  ▼                              │
│                                  ┌────────────────────────────────────────┐     │
│                                  │  UNITY 3D SCENE                        │     │
│                                  │                                        │     │
│                                  │    ○ chair     ○ lamp                  │     │
│                                  │         ○ table                        │     │
│                                  │    ○ sofa           ○ painting         │     │
│                                  │                                        │     │
│                                  │  (VLM's 3D mental model, externalized) │     │
│                                  └───────────────┬────────────────────────┘     │
│                                                  │                              │
│   GROUND TRUTH                                   │                              │
│   ────────────                                   ▼                              │
│   ┌─────────────┐                ┌────────────────────────────────────────┐     │
│   │ Spatial     │                │  EVALUATION                            │     │
│   │ Graph       │───────────────►│                                        │     │
│   │ (SAM3+DPT)  │                │  Compare VLM placements vs ground      │     │
│   └─────────────┘                │  truth spatial relationships           │     │
│                                  └────────────────────────────────────────┘     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Core Concept: Scene Program

The VLM's output is a **scene program**—a sequence of tool calls that reconstruct the spatial layout:

```python
# Zero-shot: all placements in one pass
place_object("sofa", x=-3, y=0.5, z=5)
place_object("coffee_table", x=0, y=0.5, z=3)
place_object("chair", x=4, y=0.5, z=4)

# Multi-turn: model refines after seeing each snapshot
move_object(x=-3, y=0.5, z=5, new_x=-4, new_y=0.5, new_z=6)
remove_object(x=4, y=0.5, z=4)
place_object("armchair", x=5, y=0.5, z=3)
```

## Evaluation Metrics

| Metric | Description | How Computed |
|--------|-------------|--------------|
| **Coverage** | Fraction of ground-truth entities placed | `placed / total_entities` |
| **Left/Right Accuracy** | Pairwise horizontal relations preserved | For each pair (A,B): does `A.x < B.x` match ground truth? |
| **Near/Far Accuracy** | Pairwise depth relations preserved | For each pair (A,B): does `A.z < B.z` match z_order? |

## Pipeline Components

| Component | File | Status |
|-----------|------|--------|
| **Spatial Graph Generator** | `depth_sam3_connector.py` | ✅ Complete |
| **Prompt Sources** | `prompt_sources.py` | ✅ Complete |
| **Unity Bridge** | `unity_bridge/bridge.py` | ✅ Complete |
| **Unity Tools** | `unity_bridge/tools.py` | ✅ Complete |
| **Coordinate Converter** | `spatial_graph_to_unity.py` | ✅ Complete |
| **Model Providers** | `spatial_agent/model_providers.py` | ✅ Complete |
| **System Prompts** | `spatial_agent/prompts.py` | ✅ Complete |
| **Zero-shot Eval Runner** | `run_unity_eval.py` (`--mode zero-shot`) | ✅ Complete |
| **Multi-turn Eval Runner** | `run_unity_eval.py` (`--mode multi-turn`) | ✅ Complete |
| **Experiment Runner** | `run_experiment.py` | ⬜ Not started |

---

## Implementation Phases

### Phase 1: Model Provider Abstraction ✅ Complete

Unified `VLMProvider` interface across Claude, OpenAI, HuggingFace, vLLM, and Ollama. All providers support:
- Single-turn calls (`image_path` + `prompt` + `system_prompt`)
- Multi-turn calls via `messages` (full conversation history)
- Tool calling with structured `ToolCall(name, arguments, id)` output
- `raw_assistant_message` for appending to history

### Phase 2: Evaluation Pipeline ✅ Complete

- Zero-shot mode: one API call, executes `place_object` tool calls
- Multi-turn mode: zero-shot seed → iterative snapshot + refine loop
  - Per-turn input: only the Unity snapshot (no text prompt)
  - Conversation history maintained across turns
  - Tool results fed back for proper OpenAI-spec compliance
  - All three tools available: `place_object`, `move_object`, `remove_object`
- Per-image output dirs: `unity_eval_outputs/<image_stem>/`
- Snapshot PNGs saved and listed in output JSON

### Phase 3: Experiment Runner ⬜ Not Started

**Goal**: Systematic experiments across models and images

```bash
python run_experiment.py \
    --images photos/*.jpg \
    --providers claude openai huggingface \
    --modes zero-shot multi-turn \
    --output-dir experiments/exp_001
```

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| COCO Download | ✅ Complete | `annotations/` + `val2017/` downloaded |
| Spatial Graph Pipeline | ✅ Complete | `depth_sam3_connector.py` working |
| Unity Bridge | ✅ Complete | HTTP communication + capture_view |
| Unity Scene | ✅ Complete | `CherryUnityBridge.cs` |
| Model Providers | ✅ Complete | 5 providers, conversation history support |
| Zero-shot Eval | ✅ Complete | Tested with HuggingFace/Qwen |
| Multi-turn Eval | ✅ Complete | Tested with HuggingFace/Qwen, 3 turns |
| Experiment Runner | ⬜ Not started | Batch across models/images |

---

## References

- `SPATIAL_GRAPH.md` — Ground truth generation pipeline
- `UNITY_BRIDGE.md` — Unity embodied evaluation documentation
- `unity_bridge/` — Unity communication code
- `spatial_agent/` — Model providers and prompts
- `depth_sam3_connector.py` — Spatial graph generator
