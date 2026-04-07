# Cherry: VLM Spatial Understanding Evaluation

A research framework for evaluating Vision-Language Models' 3D spatial understanding through embodied interaction in Unity.

## Quick Links

| Document | Description |
|----------|-------------|
| [PLAN.md](PLAN.md) | Research roadmap and implementation status |
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
│   Modes: zero-shot (single pass) · multi-turn (snapshot → refine loop)          │
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
pip install langchain langchain-core pydantic openai anthropic

# Set API keys
export ANTHROPIC_API_KEY=your_key      # for Claude
export OPENAI_API_KEY=your_key         # for GPT-4o
export HUGGINGFACE_TOKEN=hf_...        # for HuggingFace / Qwen (default)
```

## Quick Start

```bash
# Zero-shot placement (default)
python run_unity_eval.py --image photos/kitchen.jpg --provider huggingface

# Multi-turn refinement (snapshot → refine × N)
python run_unity_eval.py --image photos/kitchen.jpg \
    --provider huggingface --mode multi-turn --max-turns 3
```

## Current Status

- **Spatial Graph Pipeline**: Complete
- **Unity Bridge**: Complete
- **Model Providers**: Complete (Claude, OpenAI, HuggingFace, vLLM, Ollama)
- **Zero-shot Evaluation**: Complete
- **Multi-turn Evaluation**: Complete

## License

Research use only.
