# Cherry: VLM Spatial Understanding Evaluation

A research framework for evaluating Vision-Language Models' 3D spatial understanding through embodied interaction in Unity.

## Quick Links

| Document | Description |
|----------|-------------|
| [PLAN.md](PLAN.md) | Research roadmap and implementation phases |
| [SPATIAL_GRAPH.md](SPATIAL_GRAPH.md) | Ground truth generation pipeline (SAM3 + depth) |
| [UNITY_BRIDGE.md](UNITY_BRIDGE.md) | Unity embodied evaluation pipeline |

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│   Image ──► Spatial Graph (Ground Truth)    VLM ──► Unity Scene Program         │
│             (SAM3 + Depth)                         (place_object calls)          │
│                      │                                      │                    │
│                      └──────────► Evaluation ◄──────────────┘                    │
│                                                                                  │
│   Metrics: coverage, left/right accuracy, near/far accuracy                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/lauren-h-yoon/cherry.git
cd cherry

# Install dependencies
pip install torch torchvision transformers
pip install langchain langchain-anthropic langchain-core pydantic

# Set API keys
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key

# Optional: Download COCO dataset for ground-truth prompts
python scripts/download_coco.py --annotations --val
```

## Current Status

See [PLAN.md](PLAN.md) for detailed status. Summary:

- **Spatial Graph Pipeline**: Complete
- **Unity Bridge**: Complete
- **Model Providers**: In progress (Phase 1)
- **End-to-end Evaluation**: Blocked on model providers

## License

Research use only.
