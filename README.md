# Cherry: Z-Ordered Scene Understanding Pipeline

A pipeline for extracting z-ordered entities from images by combining SAM3 (Segment Anything Model 3) segmentation with depth estimation. Produces depth-sorted entity graphs for downstream scene understanding tasks, including an **embodied spatial intelligence evaluation system**.

## Overview

```
Image + Text Prompts
        |
+---------------------------------------------------------------+
|                  depth_sam3_connector.py                       |
|  +-------------+              +---------------------+          |
|  |   SAM3      |              |   Depth Extraction  |          |
|  | Segmentation|              |   (DPT / DINOv3)    |          |
|  +------+------+              +----------+----------+          |
|         |                                |                     |
|         +------------+-------------------+                     |
|                      |                                         |
|            +---------v---------+                               |
|            |   Z-Ordering      |                               |
|            |  (depth-based)    |                               |
|            +-------------------+                               |
+---------------------------------------------------------------+
        |
   spatial_graph.json  <-- GROUND TRUTH for evaluation
        |
+---------------------------------------------------------------+
|                    EMBODIED EVALUATION                         |
|                                                                |
|  VLM sees IMAGE --> reasons --> uses PRIMITIVE TOOLS           |
|                                        |                       |
|                                        v                       |
|                            RECORDED OUTPUTS                    |
|                    (paths, points, regions, movements)         |
|                                        |                       |
|                                        v                       |
|                           EVALUATOR compares to                |
|                           GROUND TRUTH (SAM3+Depth)            |
|                                        |                       |
|                                        v                       |
|                           EVALUATION SCORES                    |
+---------------------------------------------------------------+
```

## Key Concept: Embodied Spatial Intelligence Evaluation

This framework tests VLM spatial intelligence through **primitive tools as output mechanisms**:

1. **VLM sees only the IMAGE** - no oracle access to ground truth
2. **VLM uses TOOLS to express understanding** - draw paths, point to objects, mark regions
3. **GROUND TRUTH from SAM3+Depth** - used to EVALUATE VLM outputs, not provide them
4. **Scores computed** - by comparing VLM outputs against ground truth positions

```
+-------------------+     +-------------------+     +-------------------+
|   VLM INPUT       |     |   VLM OUTPUT      |     |   EVALUATION      |
+-------------------+     +-------------------+     +-------------------+
|                   |     |                   |     |                   |
|  Scene Image      | --> |  Tool Calls:      | --> |  Compare against  |
|  + Task Prompt    |     |  - draw_path()    |     |  SAM3+Depth       |
|                   |     |  - point_to()     |     |  ground truth     |
|  NO ground truth  |     |  - mark_region()  |     |                   |
|  provided!        |     |  - move_to()      |     |  Score: 0.0-1.0   |
|                   |     |  - rotate()       |     |                   |
+-------------------+     +-------------------+     +-------------------+
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

# Set up API keys
export ANTHROPIC_API_KEY=your_key_here    # For VLM evaluation (Claude)
export OPENAI_API_KEY=your_key_here       # For query generation (GPT-4o)
export HF_TOKEN=your_huggingface_token    # For SAM3 gated model
```

## Quick Start: Embodied Evaluation

```bash
# 1. Run perception pipeline (generates ground truth)
python depth_sam3_connector.py \
    --image scene.jpg \
    --prompts "table" "chair" "lamp" "person" \
    --output_dir spatial_outputs/

# 2. Run embodied spatial intelligence evaluation
python run_embodied_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --task suite \
    --num-tasks 20

# 3. Or run allocentric evaluation with query generation
python run_allocentric_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --mode full \
    --anchor "table"
```

---

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
```

**Outputs:**
- `{image}_masks.npz` - Binary segmentation masks
- `{image}_metadata.json` - Bounding boxes, scores, prompt info
- `{image}_viz.png` - Visualization with masks overlaid

---

### 2. `extract_depth.py` - Depth Extraction

Monocular depth estimation using DPT (default) or DINOv3.

```bash
# Single image with DPT (recommended)
python extract_depth.py --image scene.jpg --viz

# Use DINOv3 backend (requires local model setup)
python extract_depth.py --image scene.jpg --backend dinov3 --viz
```

**Outputs:**
- `{image}_depth.npy` - Depth map as numpy array
- `{image}_depth_metadata.json` - Depth statistics
- `{image}_depth_viz.png` - Side-by-side visualization

---

### 3. `depth_sam3_connector.py` - Combined Pipeline

Combines SAM3 segmentation with depth extraction to produce z-ordered spatial graphs (ground truth).

```bash
python depth_sam3_connector.py \
    --image scene.jpg \
    --prompts "table" "chair" "lamp" "person" \
    --output_dir spatial_outputs/
```

**Outputs:**
- `{image}_spatial_graph.json` - Complete spatial graph with z-ordering
- `{image}_masks.npz` - All entity masks
- `{image}_depth.npy` - Full depth map

**Spatial Graph JSON Structure (Ground Truth):**
```json
{
  "image_path": "scene.jpg",
  "image_size": [height, width],
  "nodes": [
    {
      "id": "entity_0",
      "name": "person",
      "bbox": [x1, y1, x2, y2],
      "bbox_center": [cx, cy],
      "z_order": 0,
      "relative_depth": 0.0,
      "depth_stats": {"mean": 7.68, "median": 7.52, ...}
    }
  ],
  "z_order_sequence": ["entity_0", "entity_1", ...]
}
```

---

### 4. `run_embodied_eval.py` - Embodied Spatial Intelligence Evaluation

The core evaluation script. VLM sees image, uses tools to express spatial understanding, outputs are evaluated against ground truth.

```bash
# Run full evaluation suite
python run_embodied_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --task suite \
    --num-tasks 20

# Run specific task types
python run_embodied_eval.py --graph ... --task localization --object "table"
python run_embodied_eval.py --graph ... --task path --start "person" --goal "stool"
python run_embodied_eval.py --graph ... --task egocentric --goal "lamp"
python run_embodied_eval.py --graph ... --task allocentric --reference "table" --goal "chair"
python run_embodied_eval.py --graph ... --task relation --relation-type left_of
python run_embodied_eval.py --graph ... --task perspective --reference "table" --direction "left"
```

**Task Types (5 categories):**

**Egocentric Tasks** (Camera/Image viewpoint - what the model directly sees):

| Task Type | Description | Evaluation |
|-----------|-------------|------------|
| `object_localization` | "Point to the stool" | Bbox hit (100%), category hit (50%), miss (0%) |
| `egocentric_path_planning` | "Draw a path to the stool" (START marker shown) | Start/end/validity/efficiency |
| `egocentric_spatial_qa` | "Is the table on your left?" (direction is the question) | YES/NO accuracy + pointing score |

**Allocentric Tasks** (Hypothetical viewpoint - marked position + facing direction):

| Task Type | Description | Evaluation |
|-----------|-------------|------------|
| `allocentric_path_planning` | "From marked viewpoint, draw path from A to B" | Start/end/validity/efficiency |
| `allocentric_spatial_qa` | "From marked viewpoint, is B to the left of A?" | YES/NO accuracy + pointing score |

**Key Design Principles:**
- **Egocentric**: No viewpoint marker needed - the camera/image IS the viewpoint
- **Allocentric**: Viewpoint marker shows WHERE you're standing + WHICH WAY you're facing
- **No direction hints** in path planning (model must locate objects)
- **Direction IS the question** in spatial QA (testing spatial understanding)

---

### 5. `run_allocentric_eval.py` - Allocentric Q&A Evaluation

Unified pipeline that generates allocentric queries and evaluates VLM responses.

```bash
# Step 1: Generate queries (uses GPT-4o for orientation estimation)
python run_allocentric_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --mode generate \
    --anchor "table"

# Step 2: Run evaluation on generated queries
python run_allocentric_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --mode evaluate \
    --queries queries_output/allocentric_queries.json

# Full pipeline (generate + evaluate)
python run_allocentric_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --mode full \
    --anchor "table"
```

**Query Types Generated:**
- `left_right` - "Is the lamp to the left or right of the table?"
- `front_behind` - "Is the chair in front of or behind the table?"
- `above_below` - "Is the ceiling above or below the table?"
- `visibility` - "Can you see the lamp from the table's position?"
- `nearest_object` - "What object is closest to the table?"
- `farthest_object` - "What object is farthest from the table?"
- `closest_left` - "What is the closest object to the left of the table?"

---

### 6. `generate_queries.py` - Query Generation

Generates allocentric Q&A pairs using GPT-4o for anchor orientation estimation.

```bash
python generate_queries.py \
    --input spatial_outputs/scene_spatial_graph.json \
    --output queries_output/queries.json \
    --anchor "table"
```

**Output JSON:**
```json
{
  "image_path": "scene.jpg",
  "anchor_entity": {"id": "entity_2", "name": "table"},
  "anchor_orientation": {"orientation": "facing_camera", "confidence": 0.85},
  "queries": [
    {
      "query_id": "q_001",
      "query_type": "left_right",
      "prompt": "Is the lamp to the left or right of the table?",
      "answer": "left",
      "targets": ["lamp_0"]
    }
  ]
}
```

---

## Primitive Tools (Output Mechanisms)

VLM uses these tools to express spatial understanding. All outputs are recorded and evaluated.

| Tool | Description | Evaluation |
|------|-------------|------------|
| `point_to(x, y, label)` | Point to a location | Bbox hit (100%), same-category (50%), miss (0%) |
| `draw_path(points, label)` | Draw path through scene | Start/end accuracy, lenient collision, efficiency |
| `mark_region(x1, y1, x2, y2, label)` | Mark rectangular region | IoU with ground truth bbox |
| `move_to(x, y)` | Move to position | Collision detection with objects |
| `rotate(angle)` | Rotate facing direction | Updated embodied state |
| `look_at(x, y)` | Turn to face location | Updated gaze direction |

**Evaluation Features:**

- **Partial Credit Pointing**: Score 1.0 for exact hit, 0.5 for same-category hit, 0.0 for miss
- **Bbox-based Detection**: Points are checked against object bounding boxes, not just distance
- **Lenient Collision**: Path validation uses 30% shrunk bboxes for realistic floor-level walking
- **Simplified Prompts**: No obstacle lists or coordinate hints - model reasons from image alone

**Simplified Prompt Examples:**

```
# Egocentric (no marker - camera IS viewpoint)
You are currently viewing this scene.
TASK: Draw a walking path to the leftmost lamp. Make sure to avoid obstacles.

# Allocentric (with viewpoint marker showing position + facing)
Imagine you are standing at the marked position (X) with the arrow orientation.
QUESTION: From your viewpoint, is the stool to the left of the table?
```

**Example VLM Interaction:**
```
Task: Point to the rightmost stool

VLM sees: [Scene Image]

VLM response:
I can see wooden stools around the kitchen island.
Using point_to to indicate the stool's position.

Tool call: point_to(x=3200, y=3100, label="stool")

Evaluation:
- Target: stool_0 at bbox [3400, 3300, 3700, 3700]
- VLM pointed: (3200, 3100)
- Hit stool_1 instead (same category)
- Score: 0.5 (partial credit) ◐
```

---

## Python API

```python
from spatial_agent import (
    GroundTruth,
    SpatialEvaluator,
    TaskGenerator,
    get_embodied_agent
)

# Load ground truth from SAM3+Depth pipeline
ground_truth = GroundTruth("spatial_outputs/scene_spatial_graph.json")

# Get object positions (for evaluation, not for VLM!)
table_pos = ground_truth.get_object_position("table")
print(f"Table center: {table_pos}")

# Generate tasks
generator = TaskGenerator(ground_truth)
tasks = generator.generate_all_tasks_flat(
    localization_count=5,
    path_count=5,
    egocentric_count=5,
    allocentric_count=5
)

# Run embodied agent
EmbodiedSpatialAgent, EmbodiedEvaluationRunner = get_embodied_agent()

runner = EmbodiedEvaluationRunner(
    spatial_graph_path="spatial_outputs/scene_spatial_graph.json",
    image_path="scene.jpg",
    model_name="claude-sonnet-4-20250514",
    verbose=True
)

# Run all tasks
results = runner.run_task_suite(tasks)
runner.print_summary(results)

# Access scores
print(f"Overall Score: {results['overall']['total_score']:.1%}")
for task_type, data in results['by_type'].items():
    print(f"  {task_type}: {data['average_score']:.1%}")
```

---

## File Structure

```
cherry/
+-- run_sam3.py                  # SAM3 segmentation
+-- extract_depth.py             # Depth extraction (DPT/DINOv3)
+-- depth_sam3_connector.py      # Combined pipeline -> ground truth
+-- spatial_graph.py             # Graph conversion & visualization
+-- run_embodied_eval.py         # Embodied evaluation runner
+-- run_allocentric_eval.py      # Allocentric Q&A evaluation
+-- generate_queries.py          # Query generation with GPT-4o
|
+-- spatial_agent/
|   +-- __init__.py              # Package exports
|   +-- annotator.py             # Scene annotation with waypoints
|   +-- state.py                 # Agent state management
|   +-- tools.py                 # LangChain tools (navigation)
|   +-- agent.py                 # Navigation agents
|   |
|   +-- primitive_tools.py       # OUTPUT MECHANISM tools
|   +-- ground_truth.py          # Ground truth from SAM3+Depth
|   +-- evaluator.py             # Compares VLM outputs to ground truth
|   +-- tasks.py                 # Task definitions
|   +-- embodied_agent.py        # Embodied spatial agent
|   |
|   +-- allocentric.py           # Allocentric relationship computation
|   +-- allocentric_tools.py     # Allocentric LangChain tools
|   +-- allocentric_eval.py      # Allocentric evaluation framework
|   +-- query_integration.py     # Connects queries with embodied eval
|
+-- web/                         # Web interface (FastAPI + Next.js)
|
+-- sam3/                        # SAM3 model (submodule)
+-- dinov3/                      # DINOv3 depth model (submodule)
```

---

## Model Configuration

| Component | Model | Purpose |
|-----------|-------|---------|
| VLM Evaluation | `claude-sonnet-4-20250514` | Spatial reasoning agent |
| Query Generation | GPT-4o | Anchor orientation estimation |
| Segmentation | SAM3 (facebook/sam3-hiera-large) | Object detection |
| Depth | DPT (Intel/dpt-large) | Monocular depth estimation |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for VLM evaluation (Claude) |
| `OPENAI_API_KEY` | Required for query generation (GPT-4o) |
| `HF_TOKEN` | Required for SAM3 gated model access |
| `CUDA_VISIBLE_DEVICES` | GPU selection for inference |

---

## Example Workflow

```bash
# 1. Generate ground truth from image
python depth_sam3_connector.py \
    --image scene.jpg \
    --prompts "table" "chair" "lamp" "person" \
    --output_dir spatial_outputs/

# 2. Run full embodied evaluation
python run_embodied_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --task suite \
    --num-tasks 30 \
    --output-dir eval_results/

# 3. Or run allocentric Q&A evaluation
python run_allocentric_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --mode full \
    --anchor "table" \
    --output-dir allocentric_results/
```

**Expected Output:**
```
============================================================
EVALUATION SUMMARY
============================================================

Overall Score: 45.6%
Total Tasks: 4

By Task Type:
  object_localization:
    Count: 1
    Average: 50.0%   (partial credit: ◐ hit lamp_1 instead of lamp_0)
  egocentric_path_planning:
    Count: 1
    Average: 52.5%   (direction correct, 1 backtrack)
  allocentric_path_planning:
    Count: 1
    Average: 30.1%   (start/end incorrect, 1 collision)
  egocentric_spatial_qa:
    Count: 1
    Average: 50.0%   (YES/NO correct, pointing missed)
  allocentric_spatial_qa:
    Count: 1
    Average: 50.0%   (YES/NO correct, pointing missed)

============================================================
DETAILED RESULTS
============================================================

Task task_0001 (object_localization):
  ◐ Pointing to lamp_0: error=124px, score=50% (hit lamp_1 instead)

Task task_0002 (egocentric_path_planning):
  ✓ Start correct, ✗ End incorrect
  ✓ Path valid (lenient mode), 1 collision avoided
  Direction: backward (correct)

Task task_0004 (egocentric_spatial_qa):
  ✓ YES/NO: VLM=NO, GT=NO (correct)
  ✗ Pointing: error=455px, score=0%

============================================================
```

---

## Evaluation System Details

### Pointing Evaluation: Partial Credit

The pointing evaluation system awards partial credit when the VLM points to an object of the correct category but not the exact target:

| Result | Score | Symbol | Example |
|--------|-------|--------|---------|
| Exact hit | 1.0 | ✓ | Target: stool_0, VLM points inside stool_0 bbox |
| Category hit | 0.5 | ◐ | Target: stool_0, VLM points inside stool_1 bbox |
| Miss | 0.0 | ✗ | Target: stool_0, VLM points at lamp bbox |

### Path Planning: Lenient Collision Detection

Path validation uses **30% shrunk bounding boxes** for collision detection. This allows realistic floor-level walking paths that pass near objects without being penalized for "colliding" with vertical extent (table legs, chair backs, etc.).

```
Original bbox:          Shrunk bbox (30%):
┌─────────────────┐     ┌─────────────────┐
│                 │     │    ┌───────┐    │
│                 │ --> │    │ core  │    │  Path through outer
│                 │     │    │  70%  │    │  30% = OK
└─────────────────┘     │    └───────┘    │
                        └─────────────────┘
```

### Egocentric vs Allocentric Tasks

```
┌─────────────────────────────────────────────────────────────────────┐
│  EGOCENTRIC                        ALLOCENTRIC                       │
│  (Camera = Your View)              (Marked Viewpoint)                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐               ┌─────────────────┐              │
│  │                 │               │                 │              │
│  │    You see      │               │    [X]→         │              │
│  │    this scene   │               │  YOU ARE HERE   │              │
│  │                 │               │  (facing →)     │              │
│  └─────────────────┘               └─────────────────┘              │
│                                                                      │
│  No marker needed                  Marker required                   │
│                                                                      │
│  "Draw path to stool"              "From viewpoint, draw path        │
│  (no direction hint)               from table to lamp"               │
│                                                                      │
│  "Is table on left?"               "Is lamp left of table?"          │
│  (direction = question)            (relation from viewpoint)         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Egocentric** tasks use the camera/image as the viewpoint:
- Model directly sees the scene - no additional marker needed
- Path planning: "Draw a path to the stool" (model must FIND the object)
- Spatial QA: "Is the table on your left?" (direction IS the question)

**Allocentric** tasks use a hypothetical viewpoint marked on the image:
- Image annotated with viewpoint marker (position + facing direction)
- Model must mentally adopt this viewpoint
- Spatial relations computed in the viewpoint's reference frame
- **Facing direction changes the answer!**

**Critical Example - Same Question, Different Answers:**
```
Question: "Is the stool to the LEFT of the table?"

Object positions: stool.x (3552) > table.x (2832)
In image coordinates: stool is to the RIGHT of table

Case A: Facing CAMERA (↑)     → Answer: NO  (standard view)
Case B: Facing AWAY (↓)       → Answer: YES (180° flip inverts left/right)
Case C: Facing LEFT (←)       → Answer: NO  (different reference axis)

This tests TRUE allocentric reasoning - mental rotation required!
```

### Z-Order Interpretation

The depth pipeline assigns z-order values where:
- **z_order = 0** → Background (farthest from camera)
- **z_order = N** → Foreground (closest to camera)
- Higher z-order = visually in front

---

## License

Research use only. See individual model licenses for SAM3 and DPT/DINOv3.
