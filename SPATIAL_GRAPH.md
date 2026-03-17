# Spatial Graph: Depth + Segmentation Pipeline

This document describes the spatial graph generation pipeline that combines SAM3 segmentation with depth estimation to produce z-ordered entity graphs.

## Overview

The spatial graph pipeline extracts objects from images and computes their spatial relationships (especially depth/z-order) for use as ground truth in spatial reasoning evaluation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SPATIAL GRAPH PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT                     PROCESSING                      OUTPUT           │
│  ─────                     ──────────                      ──────           │
│                                                                             │
│  ┌─────────────┐     ┌─────────────────────┐     ┌─────────────────────┐   │
│  │   Image     │     │  SAM3 Segmentation  │     │  spatial_graph.json │   │
│  │   (.jpg)    │────►│  (text-prompted)    │────►│  - entities         │   │
│  └─────────────┘     └─────────────────────┘     │  - z-order          │   │
│                              │                   │  - bboxes           │   │
│  ┌─────────────┐             │                   │  - depth stats      │   │
│  │   Prompts   │─────────────┘                   └─────────────────────┘   │
│  │ "chair"     │                                                            │
│  │ "table"     │     ┌─────────────────────┐     ┌─────────────────────┐   │
│  │ "lamp"      │     │  Depth Estimation   │     │  Visualization      │   │
│  └─────────────┘     │  (DPT / DINOv3)     │────►│  _spatial_viz.png   │   │
│                      └─────────────────────┘     └─────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Preparation

### COCO Dataset (Optional)

The COCO dataset is **only required** if you want to use the `coco_gt` prompt source, which looks up ground-truth object annotations for COCO images. For your own photos, use `gpt4o` or `vocabulary` prompt sources instead.

**Download COCO 2017:**

```bash
# Download annotations and validation images (recommended for dev, ~1.3GB total)
python scripts/download_coco.py --annotations --val

# Download everything including training images (~19GB total)
python scripts/download_coco.py --all

# List available files
python scripts/download_coco.py --list
```

**Directory structure after download:**

```
cherry/
├── annotations/
│   ├── instances_val2017.json      ← Object annotations (for coco_gt)
│   ├── instances_train2017.json
│   ├── captions_val2017.json       ← Not used by pipeline
│   └── ...
├── val2017/                        ← 5,000 validation images
│   ├── 000000000139.jpg
│   └── ...
└── train2017/                      ← 118,287 training images (optional)
```

**Which annotation file to use:**

| Image source | Annotation file |
|--------------|-----------------|
| `val2017/*.jpg` | `annotations/instances_val2017.json` |
| `train2017/*.jpg` | `annotations/instances_train2017.json` |

**Example with COCO ground-truth:**

```bash
# Use actual objects annotated in COCO for this specific image
python depth_sam3_connector.py \
    --image val2017/000000397133.jpg \
    --prompt-source coco_gt \
    --coco-annotations annotations/instances_val2017.json
```

### Custom Images

For your own photos (not from COCO), use one of these prompt sources:

```bash
# Auto-detect objects with GPT-4o (requires OPENAI_API_KEY)
python depth_sam3_connector.py \
    --image photos/my_room.jpg \
    --prompt-source gpt4o

# Use COCO vocabulary for indoor scenes (no API needed)
python depth_sam3_connector.py \
    --image photos/my_room.jpg \
    --prompt-source vocabulary \
    --scene-type indoor
```

## Pipeline Stages

### Stage 1: Prompt Generation

Prompts tell SAM3 what objects to look for. Four sources available:

| Source | Description | Use Case |
|--------|-------------|----------|
| `manual` | User-provided list | Known objects |
| `vocabulary` | COCO category vocabulary | General indoor/outdoor scenes |
| `coco_gt` | COCO ground-truth annotations | COCO dataset images |
| `gpt4o` | GPT-4o auto-detection | Unknown scenes |

### Stage 2: SAM3 Segmentation

Text-prompted segmentation using SAM3:
- Input: Image + text prompts
- Output: Masks + bounding boxes + confidence scores
- Each prompt may yield multiple instances (e.g., "chair" → chair_0, chair_1)

### Stage 3: Depth Estimation

Monocular depth estimation for z-ordering:

| Backend | Model | Notes |
|---------|-------|-------|
| `dpt` | Intel/dpt-large | Default, good quality |
| `dinov3` | DINOv2 + DPT head | Alternative |

### Stage 4: Filtering

Three-stage filtering to control output quality:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FILTERING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RAW DETECTIONS                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ chair_0 (0.95), chair_1 (0.82), chair_2 (0.31), chair_3 (0.15)      │   │
│  │ table_0 (0.88), lamp_0 (0.72), lamp_1 (0.45), tiny_obj (0.90)       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  STEP 1: Confidence Filter (--min_confidence 0.5)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ chair_0 (0.95), chair_1 (0.82), table_0 (0.88), lamp_0 (0.72)       │   │
│  │ tiny_obj (0.90)                                                      │   │
│  │ [Removed: chair_2, chair_3, lamp_1 - below threshold]               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  STEP 2: Area Filter (--min_area_ratio 0.01)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ chair_0 (0.95), chair_1 (0.82), table_0 (0.88), lamp_0 (0.72)       │   │
│  │ [Removed: tiny_obj - bbox < 1% of image area]                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  STEP 3: Per-Category Limit (--max_per_category 2)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ chair_0 (0.95), chair_1 (0.82)  [kept top 2 chairs by confidence]   │   │
│  │ table_0 (0.88), lamp_0 (0.72)                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  FINAL: 4 entities (filtered from 8 raw detections)                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 5: Z-Order Computation

Entities sorted by median depth (closest to camera = z_order 0):

```
z_order=0  painting_1   (depth: 6.53)   ← closest to camera
z_order=1  painting_0   (depth: 6.65)
z_order=2  table_2      (depth: 8.00)
z_order=3  chair_0      (depth: 10.63)
z_order=4  sofa_0       (depth: 17.99)  ← farthest from camera
```

## Output Format

### spatial_graph.json

```json
{
  "image_path": "photos/living_room.jpg",
  "image_size": [391, 500],
  "nodes": [
    {
      "id": "entity_0",
      "name": "chair_0",
      "category": "chair",
      "bbox": [287.6, 230.5, 355.8, 318.9],
      "bbox_center": [321.7, 274.7],
      "confidence": 0.97,
      "z_order": 5,
      "relative_depth": 0.21,
      "depth_stats": {
        "mean": 10.67,
        "min": 6.41,
        "max": 12.40,
        "std": 0.94,
        "median": 10.63,
        "pixel_count": 4341
      }
    }
  ],
  "z_order_sequence": ["entity_9", "entity_8", "entity_5", ...],
  "metadata": {
    "num_entities": 14,
    "depth_backend": "dpt",
    "prompts": ["chair", "table", "lamp", "painting", "vase", "sofa"],
    "timestamp": "2026-03-14T03:07:05"
  }
}
```

### Output Files

| File | Description |
|------|-------------|
| `{name}_spatial_graph.json` | Entity graph with z-ordering |
| `{name}_spatial_viz.png` | Visualization with bboxes and depth map |
| `{name}_masks.npz` | Binary masks for each entity (optional) |
| `{name}_depth.npy` | Full depth map (optional) |

## Usage

### Basic Usage

```bash
# Manual prompts
python depth_sam3_connector.py \
    --image photos/living_room.jpg \
    --prompts "chair" "table" "sofa" "lamp" "painting"

# Auto-detect with GPT-4o
python depth_sam3_connector.py \
    --image photos/kitchen.jpg \
    --prompt-source gpt4o

# COCO vocabulary for indoor scenes
python depth_sam3_connector.py \
    --image photos/bedroom.jpg \
    --prompt-source vocabulary \
    --scene-type indoor
```

### With Filtering

```bash
# Limit detections for cleaner output
python depth_sam3_connector.py \
    --image photos/living_room.jpg \
    --prompt-source gpt4o \
    --max_per_category 3 \
    --min_confidence 0.5 \
    --min_area_ratio 0.01
```

### CLI Options

**Input:**

| Option | Description |
|--------|-------------|
| `--image, -i` | Input image path (required) |
| `--prompts, -p` | Text prompts (space-separated) |
| `--prompts-file` | JSON file with prompts |
| `--prompt-source` | `manual`, `vocabulary`, `coco_gt`, `gpt4o` |
| `--scene-type` | For vocabulary: `indoor`, `outdoor`, `kitchen`, etc. |
| `--coco-annotations` | COCO instances JSON (for coco_gt) |

**Filtering:**

| Option | Default | Description |
|--------|---------|-------------|
| `--max_per_category` | 0 (unlimited) | Max instances per category |
| `--min_confidence` | 0.0 | Post-detection confidence threshold |
| `--min_area_ratio` | 0.0 | Min bbox area as ratio of image |

**Model:**

| Option | Default | Description |
|--------|---------|-------------|
| `--sam3_confidence` | 0.3 | SAM3 initial confidence threshold |
| `--depth_backend` | dpt | `dpt` or `dinov3` |
| `--device` | cuda | `cuda` or `cpu` |

**Output:**

| Option | Description |
|--------|-------------|
| `--output_dir, -o` | Output directory (default: spatial_outputs) |
| `--no_viz` | Skip visualization |
| `--no_masks` | Skip saving masks |
| `--no_depth` | Skip saving depth map |

## Integration with Unity Pipeline

The spatial graph serves as **ground truth** for evaluating VLM spatial reasoning:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  SPATIAL GRAPH (Ground Truth)          UNITY (VLM Output)                  │
│  ────────────────────────────          ──────────────────                  │
│                                                                             │
│  nodes[i].bbox_center[0]      ←──►     placed_object.x                     │
│  (image X pixel)                       (Unity X coordinate)                 │
│                                                                             │
│  nodes[i].z_order             ←──►     placed_object.z                     │
│  (0 = closest)                         (lower Z = closer)                   │
│                                                                             │
│  EVALUATION:                                                                │
│  - left_right_accuracy: pairwise X comparisons match?                      │
│  - near_far_accuracy: pairwise z_order comparisons match?                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Use `spatial_graph_to_unity.py` to convert spatial graph coordinates to Unity space:

```python
from spatial_graph_to_unity import convert_graph_file

unity_entities = convert_graph_file("spatial_outputs/scene_spatial_graph.json")
# Returns list of {label, x, y, z, scale} for Unity placement
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No objects detected | Lower `--sam3_confidence` or check prompts match image content |
| Too many duplicates | Use `--max_per_category` to limit instances |
| Tiny objects cluttering | Use `--min_area_ratio 0.01` to filter small detections |
| Depth looks wrong | Try `--depth_backend dinov3` as alternative |
| Out of GPU memory | Use `--device cpu` (slower) or reduce image size |
