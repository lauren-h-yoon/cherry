# Cherry: Z-Ordered Scene Understanding Pipeline

A pipeline for extracting z-ordered entities from images by combining SAM3 (Segment Anything Model 3) segmentation with depth estimation. Produces depth-sorted entity graphs for downstream scene understanding tasks, including an **agentic spatial reasoning evaluation system**.

## Overview

```
Image + Text Prompts
        ↓
┌───────────────────────────────────────────────────────────┐
│                  depth_sam3_connector.py                  │
│  ┌─────────────┐              ┌─────────────────────┐    │
│  │   SAM3      │              │   Depth Extraction  │    │
│  │ Segmentation│              │   (DPT / DINOv3)    │    │
│  └──────┬──────┘              └──────────┬──────────┘    │
│         │                                │               │
│         └────────────┬───────────────────┘               │
│                      ↓                                   │
│            ┌─────────────────┐                           │
│            │   Z-Ordering    │                           │
│            │  (depth-based)  │                           │
│            └─────────────────┘                           │
└───────────────────────────────────────────────────────────┘
        ↓
   spatial_graph.json
        ↓
┌───────────────────────────────────────────────────────────┐
│                    spatial_graph.py                       │
│  • NetworkX graph conversion                              │
│  • Z-order visualization layouts                          │
│  • Depth-based queries (closer/farther)                   │
│  • Export (GraphML, GEXF, JSON)                          │
└───────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────┐
│                    spatial_agent/                         │
│  • Agentic spatial reasoning evaluation                   │
│  • LangChain/LangGraph tools (waypoints, navigation)      │
│  • Multimodal VLM integration                             │
│  • Occlusion reasoning benchmark                          │
└───────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

```bash
# Create conda environment
conda create -n cherry python=3.10
conda activate cherry

# Core dependencies
pip install torch torchvision
pip install transformers  # For DPT depth model
pip install networkx      # For graph operations
pip install matplotlib numpy pillow

# For SAM3 (additional)
pip install iopath timm pycocotools einops decord psutil ftfy regex

# For spatial_agent (LangChain-based agent)
pip install langchain langchain-anthropic langchain-core langgraph anthropic pydantic
```

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/cherry.git
cd cherry

# Initialize submodules (SAM3 and depth models)
git submodule update --init --recursive

# Set up API keys for agent (optional)
export ANTHROPIC_API_KEY=your_key_here
export HF_TOKEN=your_huggingface_token  # For SAM3 gated model
```

## Scripts

### 1. `run_sam3.py` - SAM3 Segmentation

Text-prompted image segmentation using SAM3.

```bash
# Single image with one prompt
python run_sam3.py --image scene.jpg --prompt "table"

# Multiple prompts
python run_sam3.py --image scene.jpg --prompts "table" "chair" "lamp" "person"

# Batch processing from VLM results JSON
python run_sam3.py --batch vlm_results.json --output_dir outputs/

# Without visualization
python run_sam3.py --image scene.jpg --prompts "table" "chair" --no_viz
```

**Outputs:**
- `{image}_masks.npz` - Binary segmentation masks
- `{image}_metadata.json` - Bounding boxes, scores, prompt info
- `{image}_viz.png` - Visualization with masks overlaid

---

### 2. `extract_depth.py` - Depth Extraction

Monocular depth estimation using DPT (default) or DINOv3.

```bash
# Single image with DPT (recommended, easy setup)
python extract_depth.py --image scene.jpg --viz

# Multiple images
python extract_depth.py --images img1.jpg img2.jpg --output_dir depths/

# Use DINOv3 backend (requires local model setup)
python extract_depth.py --image scene.jpg --backend dinov3 --viz

# Process directory
python extract_depth.py --input_dir ./images/ --output_dir depths/ --viz
```

**Outputs:**
- `{image}_depth.npy` - Depth map as numpy array
- `{image}_depth.pt` - Depth map as PyTorch tensor
- `{image}_depth_metadata.json` - Depth statistics
- `{image}_depth_viz.png` - Side-by-side visualization

---

### 3. `depth_sam3_connector.py` - Combined Pipeline

Combines SAM3 segmentation with depth extraction to produce z-ordered spatial graphs.

```bash
# Analyze scene with multiple prompts
python depth_sam3_connector.py \
    --image scene.jpg \
    --prompts "table" "chair" "lamp" "person" \
    --output_dir spatial_outputs/

# Skip visualization
python depth_sam3_connector.py \
    --image scene.jpg \
    --prompts "furniture" "person" \
    --no_viz

# Use DINOv3 for depth (if available)
python depth_sam3_connector.py \
    --image scene.jpg \
    --prompts "table" "chair" \
    --depth_backend dinov3
```

**Outputs:**
- `{image}_spatial_graph.json` - Complete spatial graph with z-ordering
- `{image}_masks.npz` - All entity masks
- `{image}_depth.npy` - Full depth map
- `{image}_spatial_viz.png` - Visualization with z-ordering

**Spatial Graph JSON Structure:**
```json
{
  "image_path": "scene.jpg",
  "image_size": [height, width],
  "nodes": [
    {
      "id": "entity_0",
      "name": "person",
      "category": "person",
      "bbox": [x1, y1, x2, y2],
      "bbox_center": [cx, cy],
      "confidence": 0.96,
      "z_order": 0,
      "relative_depth": 0.0,
      "depth_stats": {
        "mean": 7.68,
        "min": 5.43,
        "max": 9.87,
        "std": 0.73,
        "median": 7.52,
        "pixel_count": 259026
      }
    }
  ],
  "z_order_sequence": ["entity_0", "entity_1", ...],
  "metadata": {...}
}
```

**Z-Order:**
- `z_order=0`: Closest to camera
- `z_order=N`: Farthest from camera
- `relative_depth`: Normalized depth (0.0 = closest, 1.0 = farthest)

---

### 4. `spatial_graph.py` - Z-Order Visualization

Converts spatial graph JSON to NetworkX and provides z-order visualization.

```bash
# Generate visualization (hierarchical layout - default)
python spatial_graph.py \
    --input spatial_graph.json \
    --viz \
    --output graph.png

# Different layout types
python spatial_graph.py -i graph.json --viz --viz_type depth_layers
python spatial_graph.py -i graph.json --viz --viz_type hierarchical
python spatial_graph.py -i graph.json --viz --viz_type circular

# Query entities by depth
python spatial_graph.py -i graph.json --query closer --entity table_0
python spatial_graph.py -i graph.json --query farther --entity person

# Export to other formats
python spatial_graph.py -i graph.json --export graphml --export_path scene.graphml
python spatial_graph.py -i graph.json --export gexf --export_path scene.gexf
```

**Visualization Types:**
| Type | Description |
|------|-------------|
| `hierarchical` | Z-ordered layers (top=close, bottom=far) |
| `depth_layers` | X from image position, Y from depth |
| `circular` | Circular arrangement |
| `spring` | Force-directed layout |

---

### 5. `spatial_agent/` - Agentic Spatial Reasoning

An evaluation framework for testing VLM/agent spatial reasoning capabilities using depth-extracted waypoints.

#### Concept

The spatial agent system tests whether AI agents can:
- Understand z-order relationships from visual input
- Reason about occlusion (objects blocking paths)
- Navigate through 3D space using waypoints
- Plan valid paths that respect depth ordering

```
┌─────────────────────────────────────────────────────────────┐
│                    Annotated Scene                          │
│                                                             │
│   ▲ AGENT (z=0)     Waypoints colored by depth:            │
│   ● z=1             • Green = near camera                   │
│   ● z=2             • Red = far from camera                 │
│   ● z=3                                                     │
│   ★ TARGET (z=4)    Agent must navigate to target          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Tools Available to Agent

| Tool | Description |
|------|-------------|
| `get_waypoints` | List all waypoints with z-order and positions |
| `move_to` | Navigate to a specific waypoint |
| `rotate` | Rotate view to understand spatial relationships |
| `scale` | Zoom in/out to see depth relationships |

#### Usage

```bash
# Run spatial agent evaluation
python run_spatial_agent.py \
    --graph spatial_outputs/4r9OKorlcTk_spatial_graph.json \
    --agent-z 0 \
    --target-z 8 \
    --output-dir spatial_agent_outputs/

# With multimodal input (image + text)
python run_spatial_agent.py \
    --graph spatial_outputs/4r9OKorlcTk_spatial_graph.json \
    --multimodal

# Custom start/target positions
python run_spatial_agent.py \
    --graph spatial_graph.json \
    --agent-z 2 \
    --target-z 6
```

#### Python API

```python
from spatial_agent import SpatialAnnotator, get_agent

# Generate annotated scene
annotator = SpatialAnnotator("spatial_graph.json")
config = annotator.create_scene_config(agent_z=0, target_z=8)
annotator.annotate(config, "annotated_scene.png")

# Run agent evaluation
SpatialReasoningAgent = get_agent()
agent = SpatialReasoningAgent(verbose=True)

agent.setup_scenario(
    spatial_graph_path="spatial_graph.json",
    agent_z=0,
    target_z=8
)

result = agent.run_with_image()
print(f"Reached target: {result['reached_target']}")
print(f"Path taken: {result['path_summary']}")
```

#### Output

The agent produces:
- `scenario_annotated.png` - Annotated scene with waypoints
- Conversation log with reasoning steps
- Path summary and success/failure status

```
=== Results ===
Reached target: True
Total moves: 4
Path: entity_8 → entity_5 → entity_1 → entity_0 → entity_4
```

---

## Example Workflow

```bash
# 1. Run the full pipeline
python depth_sam3_connector.py \
    --image visual_scenes_2d_unsplash/architecture-interior/4r9OKorlcTk.jpg \
    --prompts "table" "stool" "lamp" "person" \
    --output_dir spatial_outputs/

# 2. Visualize the spatial graph
python spatial_graph.py \
    --input spatial_outputs/4r9OKorlcTk_spatial_graph.json \
    --viz \
    --viz_type hierarchical \
    --output spatial_outputs/graph_viz.png

# 3. Run spatial reasoning agent
python run_spatial_agent.py \
    --graph spatial_outputs/4r9OKorlcTk_spatial_graph.json \
    --multimodal \
    --output-dir spatial_agent_outputs/

# 4. Export for external tools
python spatial_graph.py \
    --input spatial_outputs/4r9OKorlcTk_spatial_graph.json \
    --export graphml
```

## Output Examples

### Z-Ordering (Front to Back)
```
z=0: person (depth=0.000)      ← Closest to camera
z=1: lamp_0 (depth=0.233)
z=2: table_1 (depth=0.268)
z=3: lamp_2 (depth=0.562)
z=4: lamp_1 (depth=0.566)
z=5: table_0 (depth=0.636)
z=6: stool_1 (depth=0.669)
z=7: stool_0 (depth=0.752)
z=8: stool_2 (depth=1.000)     ← Farthest from camera
```

## File Structure

```
cherry/
├── run_sam3.py              # SAM3 segmentation
├── extract_depth.py         # Depth extraction (DPT/DINOv3)
├── depth_sam3_connector.py  # Combined pipeline
├── spatial_graph.py         # Graph conversion & visualization
├── run_spatial_agent.py     # Agent evaluation runner
├── requirements.txt         # Python dependencies
│
├── spatial_agent/           # Agentic spatial reasoning module
│   ├── __init__.py
│   ├── agent.py             # LangGraph-based agent
│   ├── annotator.py         # Scene annotation with waypoints
│   ├── state.py             # Agent state management
│   └── tools.py             # LangChain tools
│
├── sam3/                    # SAM3 model (submodule)
├── dinov3/                  # DINOv3 depth model (submodule)
│
├── visual_scenes_2d_unsplash/  # Sample images
├── spatial_outputs/            # Pipeline outputs
└── spatial_agent_outputs/      # Agent evaluation outputs
```

## API Usage

```python
from depth_sam3_connector import DepthSAM3Connector
from spatial_graph import SpatialSceneGraph
from spatial_agent import SpatialAnnotator, AgentState, get_agent, get_tools
from PIL import Image

# === Pipeline ===
connector = DepthSAM3Connector(depth_backend="dpt", device="cuda")
image = Image.open("scene.jpg")
graph = connector.analyze(image, prompts=["table", "chair", "person"])

for node in graph.nodes:
    print(f"{node.name}: z_order={node.z_order}, depth={node.relative_depth:.2f}")

# === Graph Queries ===
sg = SpatialSceneGraph.from_json("spatial_graph.json")
z_ordered = sg.get_z_ordered_entities()
closer = sg.get_entities_closer_than("entity_3")
farther = sg.get_entities_farther_than("entity_0")
sg.visualize(output_path="graph.png", viz_type="hierarchical")

# === Spatial Agent ===
annotator = SpatialAnnotator("spatial_graph.json")
config = annotator.create_scene_config(agent_z=0, target_z=8)
annotator.annotate(config, "scene_annotated.png")

# Create and run agent
SpatialReasoningAgent = get_agent()
agent = SpatialReasoningAgent()
agent.setup_scenario("spatial_graph.json", agent_z=0, target_z=8)
result = agent.run_with_image()
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for spatial_agent (Claude API) |
| `HF_TOKEN` | Required for SAM3 gated model access |
| `CUDA_VISIBLE_DEVICES` | GPU selection for inference |

## Notes

- **SAM3 prompts**: Works best with short, simple prompts ("table", "chair") rather than long descriptions
- **Depth backend**: DPT is recommended for ease of use; DINOv3 requires local model weights
- **GPU**: CUDA recommended for reasonable performance
- **Conda environment**: Use `cherry` environment for reproducibility across local/remote

## License

Research use only. See individual model licenses for SAM3 and DPT/DINOv3.
