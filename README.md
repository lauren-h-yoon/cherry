# Cherry: Z-Ordered Scene Understanding Pipeline

A pipeline for extracting z-ordered entities from images by combining SAM3 (Segment Anything Model 3) segmentation with depth estimation. It produces depth-sorted spatial graphs for downstream evaluation, including:

- the current **query benchmark** workflow
- the older **embodied spatial evaluation** workflow

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

## Current Workflows

Cherry currently has two evaluation paths:

1. **Query benchmark** (current primary path)
   - Stage 1 builds `spatial_graph.json` and a labeled `*_spatial_viz.png`
   - models answer egocentric and allocentric spatial queries directly
   - outputs are saved per image and model under `query_benchmark_outputs/`

2. **Embodied evaluation** (older path, still available)
   - models use primitive tools like `point_to()` and `draw_path()`
   - outputs are evaluated against the same Stage 1 ground truth

## Key Concept: Query Benchmark

The current benchmark is designed to isolate spatial reasoning more cleanly than the primitive-tool setup.

1. **Stage 1 builds the scene representation**
   - SAM3 + depth produce `spatial_graph.json`
   - Stage 1 also produces a labeled visualization image used as model input
2. **Model sees the labeled image**
   - object boxes and labels are already shown
   - the model answers with a text label only, such as `left`, `behind`, or `chair_0`
3. **Evaluation is exact-match on the normalized answer**
   - the benchmark compares the model answer to the ground-truth answer computed from the graph

Current query families:
- `egocentric_qa`
- `allocentric_qa`

Current query subtypes:
- `binary_relation`
- `relation_mcq`
- `object_retrieval`

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

## Quick Start: Query Benchmark

```bash
# 1. Run Stage 1 perception pipeline
python depth_sam3_connector.py \
    --image photos/living_room2.jpg \
    --prompts "chair" "lamp" "painting" "vase" \
    --output_dir spatial_outputs/

# 2. Generate query benchmark prompts only
python run_query_benchmark.py \
    --graph spatial_outputs/living_room2_spatial_graph.json \
    --output-dir query_benchmark_outputs/

# 3. Run and evaluate one model
python run_query_benchmark.py \
    --graph spatial_outputs/living_room2_spatial_graph.json \
    --render-image \
    --evaluate \
    --provider openai \
    --model gpt-4o \
    --output-dir query_benchmark_outputs/

# 4. Analyze results
python analyze_query_results.py \
    --input query_benchmark_outputs/living_room2__openai__gpt-4o/evaluation_results.json
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

```

## Batch Scripts

For repeated runs across many images and models, use the shell helpers under `scripts/`.

```bash
# Query benchmark batch run
chmod +x scripts/run_query_benchmark_batch.sh
MODEL_FILTER=openai,claude ./scripts/run_query_benchmark_batch.sh

# Stage 1 spatial graph generation batch helper
chmod +x scripts/generate_all_spatial_graphs.sh
./scripts/generate_all_spatial_graphs.sh

# Older embodied-evaluation batch helper
chmod +x scripts/run_all_evaluations.sh
./scripts/run_all_evaluations.sh
```

Useful query-batch filters:
- `IMAGE_FILTER=living_room2`
- `IMAGE_LIMIT=1`
- `MODEL_FILTER=openai,claude`

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

Convention:
- `z_order = 0` is farthest from the camera / background
- larger `z_order` values are closer to the camera / foreground
- `z_order_sequence` is ordered from back to front

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

### 4. `run_query_benchmark.py` - Query Benchmark Runner

Generate and optionally evaluate text-only spatial queries using the Stage 1 labeled scene image.

```bash
# Generate queries only
python run_query_benchmark.py \
    --graph spatial_outputs/living_room2_spatial_graph.json \
    --output-dir query_benchmark_outputs/

# Evaluate with GPT-4o
python run_query_benchmark.py \
    --graph spatial_outputs/living_room2_spatial_graph.json \
    --render-image \
    --evaluate \
    --provider openai \
    --model gpt-4o \
    --output-dir query_benchmark_outputs/
```

**Outputs:**
- `<image>__queries/queries.json`
- `<image>__queries/queries_readable.txt`
- `<image>__<provider>__<model>/evaluation_results.json`
- model-specific analysis files produced by `analyze_query_results.py`

---

### 5. `analyze_query_results.py` - Query Benchmark Analysis

Summarize query benchmark results and save text/JSON reports plus plots.

```bash
python analyze_query_results.py \
    --input query_benchmark_outputs/living_room2__openai__gpt-4o/evaluation_results.json
```

**Outputs:**
- `analysis_summary.txt`
- `analysis_summary.json`
- accuracy plots by task type, frame type, subtype, relation axis, and template
- `task_frame_subtype_table.png`

---

### 6. `run_embodied_eval.py` - Embodied Spatial Intelligence Evaluation

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

### 7. Legacy Scripts

Older scripts are kept under `legacy/` for reference:

- `legacy/generate_queries.py`
- `legacy/visualize_tasks.py`
- `legacy/run_spatial_agent.py`

Older ad-hoc output folders are also kept there when retained for reference.

The query benchmark does **not** use `legacy/generate_queries.py` for query generation. That script is only still reused for Stage 1 prompt detection when object prompts need to be auto-generated from an image.

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
+-- depth_sam3_connector.py      # Stage 1: SAM3 + depth -> spatial graph
+-- run_sam3.py                  # SAM3 segmentation helper
+-- extract_depth.py             # Depth extraction helper
+-- run_query_benchmark.py       # Current query benchmark runner
+-- analyze_query_results.py     # Query benchmark analysis
+-- run_embodied_eval.py         # Older embodied evaluation runner
+-- query_benchmark/             # Query schema, templates, ground truth, generator
+-- spatial_agent/               # Embodied evaluation stack + model providers
+-- scripts/                     # Batch scripts
+-- legacy/                      # Older scripts kept for reference
+-- spatial_outputs/             # Stage 1 outputs
+-- query_benchmark_outputs/     # Query benchmark outputs
+-- sam3/                        # SAM3 model (submodule)
+-- dinov3/                      # DINOv3 depth model (submodule)
```

---

## Model Configuration

| Component | Model | Purpose |
|-----------|-------|---------|
| Query Benchmark Evaluation | `gpt-4o`, `claude-sonnet-4-5`, `vllm`/Qwen | Text-only spatial QA |
| Embodied Evaluation | Claude / OpenAI / Qwen / Ollama | Primitive-tool benchmark |
| Stage 1 Prompt Detection | GPT-4o via `legacy/generate_queries.py` | Auto-detect object prompts |
| Segmentation | SAM3 | Object detection / masks |
| Depth | DPT or DINOv3 | Monocular depth estimation |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for Claude evaluation |
| `OPENAI_API_KEY` | Required for OpenAI evaluation and Stage 1 prompt detection |
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

# 2. Run query benchmark evaluation
python run_query_benchmark.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --render-image \
    --evaluate \
    --provider openai \
    --model gpt-4o \
    --output-dir query_benchmark_outputs/

# 3. Analyze query benchmark outputs
python analyze_query_results.py \
    --input query_benchmark_outputs/scene__openai__gpt-4o/evaluation_results.json

# 4. Or run the older embodied benchmark
python run_embodied_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --task suite \
    --num-tasks 30 \
    --output-dir embodied_eval_outputs/
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
