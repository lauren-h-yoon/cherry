# Cherry Pipeline Flow

This document provides a comprehensive end-to-end walkthrough of the Cherry pipeline, from raw image input to embodied spatial intelligence evaluation.

---

## Executive Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EMBODIED SPATIAL INTELLIGENCE PIPELINE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌───────────────────┐    ┌─────────────────────────┐  │
│  │   INPUT      │    │   GROUND TRUTH    │    │      VLM AGENT          │  │
│  │   IMAGE      │───►│   (SAM3 + Depth)  │    │ (Qwen3-VL / Claude /    │  │
│  └──────────────┘    │                   │    │  GPT-4o / Ollama)       │  │
│                      │  • Object bboxes  │    │                         │  │
│                      │  • Z-order        │    │   Sees image + prompt   │  │
│                      │  • Spatial rels   │    │         │               │  │
│                      └─────────┬─────────┘    │         ▼               │  │
│                                │              │   Uses PRIMITIVE TOOLS  │  │
│                                │              │   as OUTPUT MECHANISMS  │  │
│                                │              └───────────┬─────────────┘  │
│                                │                          │                │
│                                ▼                          ▼                │
│                      ┌─────────────────────────────────────────────────┐   │
│                      │              EVALUATOR                           │   │
│                      │   Compare VLM outputs vs Ground Truth            │   │
│                      │                                                  │   │
│                      │   Metrics:                                       │   │
│                      │   • Pointing: bbox hit (100%) or category (50%) │   │
│                      │   • Path: start + end + validity + efficiency    │   │
│                      │   • Path validity: lenient (30% bbox shrink)     │   │
│                      │   • Yes/No QA: exact match                       │   │
│                      └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Insight: Tools as Output Mechanisms

The VLM does NOT receive ground truth information. Instead:
1. VLM **sees only the image** and task prompt
2. VLM **reasons** about spatial relationships from visual input
3. VLM **expresses** understanding via primitive tools (point_to, draw_path, etc.)
4. **Evaluator** compares VLM outputs against hidden ground truth

This tests genuine spatial reasoning, not information retrieval.

---

## High-Level Architecture

```
+-----------------------------------------------------------------------------+
|                           CHERRY PIPELINE                                    |
+-----------------------------------------------------------------------------+
|                                                                              |
|  STAGE 1: PERCEPTION (Ground Truth Generation)                              |
|  +------------+    +------------+                                            |
|  |   Image    |    |   Text     |                                            |
|  |   (scene)  |    |   Prompts  |                                            |
|  +-----+------+    +-----+------+                                            |
|        |                 |                                                   |
|        +--------+--------+                                                   |
|                 v                                                            |
|  +------------------------------------------------------------+             |
|  |                  depth_sam3_connector.py                    |             |
|  |  +---------------------+    +---------------------+         |             |
|  |  |      SAM3           |    |   Depth Extractor   |         |             |
|  |  |   (Segmentation)    |    |   (DPT / DINOv3)    |         |             |
|  |  +----------+----------+    +----------+----------+         |             |
|  |             |                          |                    |             |
|  |             +------------+-------------+                    |             |
|  |                          v                                  |             |
|  |             +------------------------+                      |             |
|  |             |     Z-Order Ranking    |                      |             |
|  |             |  (median depth -> rank)|                      |             |
|  |             +------------------------+                      |             |
|  +------------------------------------------------------------+             |
|                 |                                                            |
|                 v                                                            |
|        spatial_graph.json  <-- GROUND TRUTH                                  |
|                                                                              |
+------------------------------------------------------------------------------+
|                                                                              |
|  STAGE 2: EMBODIED SPATIAL INTELLIGENCE EVALUATION                          |
|                                                                              |
|  +-----------------------------+                                             |
|  |         VLM INPUT           |                                             |
|  |  +-------+   +------------+ |                                             |
|  |  | Image |   | Task Prompt| |     NO ground truth provided!               |
|  |  +-------+   +------------+ |                                             |
|  +-------------+---------------+                                             |
|                |                                                             |
|                v                                                             |
|  +------------------------------------------------------------+             |
|  |              VLM (claude-sonnet-4-20250514)                 |             |
|  |                                                             |             |
|  |  Reasons about spatial relationships from image alone       |             |
|  +-----------------------------+-------------------------------+             |
|                |                                                             |
|                v                                                             |
|  +------------------------------------------------------------+             |
|  |              PRIMITIVE TOOLS (Output Mechanisms)            |             |
|  |                                                             |             |
|  |  point_to(x, y)     - Point to location                    |             |
|  |  draw_path(points)  - Draw navigation path                 |             |
|  |  mark_region(bbox)  - Highlight area                       |             |
|  |  move_to(x, y)      - Navigate to position                 |             |
|  |  rotate(angle)      - Change facing direction              |             |
|  |  look_at(x, y)      - Direct gaze                          |             |
|  +-----------------------------+-------------------------------+             |
|                |                                                             |
|                v                                                             |
|  +------------------------------------------------------------+             |
|  |              RECORDED OUTPUTS                               |             |
|  |                                                             |             |
|  |  paths: [(x1,y1), (x2,y2), ...]                            |             |
|  |  points: [(x, y, label), ...]                              |             |
|  |  regions: [(x1,y1,x2,y2,label), ...]                       |             |
|  |  movements: [(from, to), ...]                              |             |
|  +-----------------------------+-------------------------------+             |
|                |                                                             |
|                v                                                             |
|  +------------------------------------------------------------+             |
|  |              EVALUATOR                                      |             |
|  |                                                             |             |
|  |  Compares VLM outputs against GROUND TRUTH (SAM3+Depth)    |             |
|  |                                                             |             |
|  |  - Pointing accuracy (distance to target)                  |             |
|  |  - Path validity (no collisions)                           |             |
|  |  - Region IoU (overlap with bbox)                          |             |
|  |  - Navigation success (reached goal)                       |             |
|  +-----------------------------+-------------------------------+             |
|                |                                                             |
|                v                                                             |
|        EVALUATION SCORES (0.0 - 1.0)                                         |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

## Stage 1: Perception Pipeline (Ground Truth Generation)

### 1.1 Input Requirements

```
Inputs:
+-- Image (JPG/PNG)
|   +-- Any scene with identifiable objects
|
+-- Text Prompts (List[str])
    +-- Object categories to detect: ["table", "chair", "person", ...]
```

### 1.2 SAM3 Segmentation (`run_sam3.py`)

**Purpose:** Text-prompted instance segmentation using Segment Anything Model 3.

```
+-------------------------------------------------------------+
|                        SAM3 Pipeline                         |
+-------------------------------------------------------------+
|                                                              |
|   Image + Prompt ("table")                                   |
|           |                                                  |
|           v                                                  |
|   +-------------------+                                      |
|   |   Sam3Processor   |  (Text tokenization + Image preproc) |
|   +---------+---------+                                      |
|             |                                                |
|             v                                                |
|   +-------------------+                                      |
|   |   SAM3 Model      |  (Vision-Language Segmentation)      |
|   |   (HuggingFace)   |                                      |
|   +---------+---------+                                      |
|             |                                                |
|             v                                                |
|   +-------------------+                                      |
|   |  Post-processing  |                                      |
|   |  - NMS filtering  |                                      |
|   |  - Score threshold|                                      |
|   |  - Mask cleanup   |                                      |
|   +---------+---------+                                      |
|             |                                                |
|             v                                                |
|   Output per prompt:                                         |
|   +-- masks: np.ndarray (N, 1, H, W)  # Binary masks         |
|   +-- boxes: np.ndarray (N, 4)        # [x1, y1, x2, y2]     |
|   +-- scores: np.ndarray (N,)         # Confidence scores    |
|                                                              |
+--------------------------------------------------------------+
```

### 1.3 Depth Extraction (`extract_depth.py`)

**Purpose:** Monocular depth estimation from a single image.

```
+--------------------------------------------------------------+
|                     Depth Extraction Pipeline                 |
+--------------------------------------------------------------+
|                                                              |
|   Backend Options:                                           |
|   +---------------------+    +---------------------+         |
|   |        DPT          |    |       DINOv3        |         |
|   |  (Intel/dpt-large)  |    |  (Local weights)    |         |
|   |                     |    |                     |         |
|   |  - HuggingFace      |    |  - Requires setup   |         |
|   |  - Easy setup       |    |  - Higher quality   |         |
|   |  - Recommended      |    |  - Slower           |         |
|   +----------+----------+    +----------+----------+         |
|              |                          |                    |
|              +------------+-------------+                    |
|                           v                                  |
|              +------------------------+                      |
|              |      Image Input       |                      |
|              |      (H x W x 3)       |                      |
|              +-----------+------------+                      |
|                          |                                   |
|                          v                                   |
|              +------------------------+                      |
|              |   Depth Prediction     |                      |
|              |   (H x W) float32      |                      |
|              +------------------------+                      |
|                                                              |
+--------------------------------------------------------------+
```

### 1.4 Combined Pipeline (`depth_sam3_connector.py`)

**Purpose:** Fuse segmentation and depth to produce z-ordered spatial graph (GROUND TRUTH).

```
+--------------------------------------------------------------+
|                  Depth-SAM3 Connector Pipeline                |
+--------------------------------------------------------------+
|                                                              |
|   Step 1: Parallel Extraction                                |
|   +---------------------+    +---------------------+         |
|   |   SAM3 Segmentation |    |   Depth Extraction  |         |
|   |   For each prompt:  |    |   Full image:       |         |
|   |   -> masks, boxes   |    |   -> depth_map      |         |
|   +----------+----------+    +----------+----------+         |
|              |                          |                    |
|              +------------+-------------+                    |
|                           v                                  |
|   Step 2: Depth Statistics per Entity                        |
|   +------------------------------------------------------+   |
|   |  For each detected entity:                           |   |
|   |                                                      |   |
|   |    mask = entity.mask  # (H, W) binary               |   |
|   |    masked_depth = depth_map[mask]                    |   |
|   |                                                      |   |
|   |    stats = {                                         |   |
|   |        "mean": masked_depth.mean(),                  |   |
|   |        "median": np.median(masked_depth),  <-- Key   |   |
|   |        "min": masked_depth.min(),                    |   |
|   |        "max": masked_depth.max()                     |   |
|   |    }                                                 |   |
|   +------------------------------------------------------+   |
|                           |                                  |
|                           v                                  |
|   Step 3: Z-Order Ranking                                    |
|   +------------------------------------------------------+   |
|   |  # Sort by median depth (lower = closer)             |   |
|   |  entities.sort(key=lambda e: e.depth_median)         |   |
|   |                                                      |   |
|   |  # Assign z-order                                    |   |
|   |  for i, entity in enumerate(entities):               |   |
|   |      entity.z_order = i                              |   |
|   |                                                      |   |
|   |  # Compute relative depth [0.0, 1.0]                 |   |
|   |  entity.relative_depth = normalize(...)              |   |
|   +------------------------------------------------------+   |
|                           |                                  |
|                           v                                  |
|   Output: spatial_graph.json  <-- GROUND TRUTH               |
|                                                              |
+--------------------------------------------------------------+
```

### 1.5 Spatial Graph JSON Schema (Ground Truth)

```json
{
  "image_path": "path/to/image.jpg",
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
        "median": 7.52
      }
    }
  ],

  "z_order_sequence": ["entity_0", "entity_1", ...],

  "metadata": {
    "num_entities": 9,
    "depth_backend": "dpt",
    "prompts": ["table", "chair", "person"]
  }
}
```

---

## Stage 2: Embodied Spatial Intelligence Evaluation

### 2.1 Core Concept: Tools as Output Mechanisms

The key insight of this framework is that **tools are OUTPUT mechanisms**, not oracle data providers.

```
+--------------------------------------------------------------+
|                    EVALUATION DESIGN                          |
+--------------------------------------------------------------+
|                                                              |
|   WRONG (Oracle Tools):                                      |
|   +-------------------+     +-------------------+            |
|   |   VLM asks:       | --> |   Tool returns:   |            |
|   |   "Where is       |     |   ground truth    |            |
|   |    the table?"    |     |   position (x,y)  |            |
|   +-------------------+     +-------------------+            |
|                                                              |
|   This tests nothing - VLM just reads answers!               |
|                                                              |
+--------------------------------------------------------------+
|                                                              |
|   CORRECT (Output Mechanisms):                               |
|   +-------------------+     +-------------------+            |
|   |   VLM sees image, | --> |   VLM calls tool  |            |
|   |   reasons about   |     |   to EXPRESS its  |            |
|   |   spatial layout  |     |   understanding   |            |
|   +-------------------+     +-------------------+            |
|             |                         |                      |
|             |                         v                      |
|             |               +-------------------+            |
|             |               |   point_to(450,   |            |
|             |               |   320, "table")   |            |
|             |               +-------------------+            |
|             |                         |                      |
|             v                         v                      |
|   +-------------------+     +-------------------+            |
|   |   Ground Truth    | <-- |   EVALUATOR       |            |
|   |   table is at     |     |   compares VLM    |            |
|   |   (455, 315)      |     |   output to GT    |            |
|   +-------------------+     +-------------------+            |
|                                       |                      |
|                                       v                      |
|                             Score: 0.95 (good!)              |
|                                                              |
+--------------------------------------------------------------+
```

### 2.2 Component: `primitive_tools.py`

**Purpose:** Output mechanism tools that record VLM spatial expressions for evaluation.

```
+--------------------------------------------------------------+
|                   Primitive Tools (Output Mechanisms)         |
+--------------------------------------------------------------+
|                                                              |
|   @dataclass RecordedOutputs                                 |
|   +------------------------------------------------------+   |
|   |  paths: List[Dict]      # Drawn paths                |   |
|   |  points: List[Dict]     # Pointed locations          |   |
|   |  regions: List[Dict]    # Marked regions             |   |
|   |  movements: List[Dict]  # Movement commands          |   |
|   |  rotations: List[float] # Rotation commands          |   |
|   |  gaze: List[Dict]       # Look-at commands           |   |
|   +------------------------------------------------------+   |
|                                                              |
|   @dataclass EmbodiedState                                   |
|   +------------------------------------------------------+   |
|   |  position: Tuple[int, int]   # Current position      |   |
|   |  facing_angle: float         # Current orientation   |   |
|   +------------------------------------------------------+   |
|                                                              |
|   Tools:                                                     |
|   +------------------------------------------------------+   |
|   |  DrawPathTool                                        |   |
|   |  +-- name: "draw_path"                               |   |
|   |  +-- Input: points [(x1,y1), (x2,y2), ...], label    |   |
|   |  +-- Records: path for evaluation                    |   |
|   |                                                      |   |
|   |  PointToTool                                         |   |
|   |  +-- name: "point_to"                                |   |
|   |  +-- Input: x, y, label                              |   |
|   |  +-- Records: pointed location                       |   |
|   |                                                      |   |
|   |  MarkRegionTool                                      |   |
|   |  +-- name: "mark_region"                             |   |
|   |  +-- Input: x1, y1, x2, y2, label                    |   |
|   |  +-- Records: bounding box                           |   |
|   |                                                      |   |
|   |  MoveToTool                                          |   |
|   |  +-- name: "move_to"                                 |   |
|   |  +-- Input: x, y                                     |   |
|   |  +-- Updates: embodied state position                |   |
|   |                                                      |   |
|   |  RotateTool                                          |   |
|   |  +-- name: "rotate"                                  |   |
|   |  +-- Input: angle (degrees)                          |   |
|   |  +-- Updates: embodied state facing                  |   |
|   |                                                      |   |
|   |  LookAtTool                                          |   |
|   |  +-- name: "look_at"                                 |   |
|   |  +-- Input: x, y                                     |   |
|   |  +-- Records: gaze direction                         |   |
|   +------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
```

### 2.3 Component: `ground_truth.py`

**Purpose:** Loads ground truth from SAM3+Depth pipeline for evaluation (NOT provided to VLM).

```
+--------------------------------------------------------------+
|                   GroundTruth Class                           |
+--------------------------------------------------------------+
|                                                              |
|   Input: spatial_graph.json                                  |
|                                                              |
|   class GroundTruth:                                         |
|   +------------------------------------------------------+   |
|   |  def __init__(self, spatial_graph_path: str):        |   |
|   |      # Load spatial graph                            |   |
|   |      # Build object lookup dictionary                |   |
|   |      # Store image dimensions                        |   |
|   |                                                      |   |
|   |  def get_object_position(self, name: str)            |   |
|   |      -> Tuple[float, float]                          |   |
|   |      # Returns bbox_center of named object           |   |
|   |                                                      |   |
|   |  def get_object_bbox(self, name: str)                |   |
|   |      -> List[int]                                    |   |
|   |      # Returns [x1, y1, x2, y2]                      |   |
|   |                                                      |   |
|   |  def is_left_of(self, a: str, b: str) -> bool        |   |
|   |      # True if a.center_x < b.center_x               |   |
|   |                                                      |   |
|   |  def is_behind(self, a: str, b: str) -> bool         |   |
|   |      # True if a.z_order > b.z_order                 |   |
|   |                                                      |   |
|   |  def is_path_valid(self, path, excluded) -> bool     |   |
|   |      # Check if path collides with objects           |   |
|   |                                                      |   |
|   |  def get_view_from_position(self, pos, angle)        |   |
|   |      -> Dict[str, List[ObjectInfo]]                  |   |
|   |      # Returns objects in each direction             |   |
|   +------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
```

### 2.4 Component: `evaluator.py`

**Purpose:** Compare VLM outputs against ground truth to compute scores.

```
+--------------------------------------------------------------+
|                   SpatialEvaluator                            |
+--------------------------------------------------------------+
|                                                              |
|   class SpatialEvaluator:                                    |
|   +------------------------------------------------------+   |
|   |  def __init__(self, ground_truth: GroundTruth)       |   |
|   |                                                      |   |
|   |  def evaluate_pointing(                              |   |
|   |      vlm_point: (x, y),                              |   |
|   |      target_object: str                              |   |
|   |  ) -> PointingResult:                                |   |
|   |      # Check if point is inside target bbox (100%)   |   |
|   |      # Check if point is inside same-category (50%)  |   |
|   |      # Use 5% image diagonal as distance threshold   |   |
|   |                                                      |   |
|   |  def evaluate_path(                                  |   |
|   |      vlm_path: [(x,y), ...],                         |   |
|   |      start_object: str,                              |   |
|   |      goal_object: str                                |   |
|   |  ) -> PathResult:                                    |   |
|   |      # Check: start near start_object?               |   |
|   |      # Check: end near goal_object?                  |   |
|   |      # Check: path doesn't collide (lenient mode)    |   |
|   |      # Compute: path efficiency                      |   |
|   |                                                      |   |
|   |  def evaluate_yes_no(                                |   |
|   |      vlm_response: str,                              |   |
|   |      ground_truth_answer: bool                       |   |
|   |  ) -> YesNoResult:                                   |   |
|   |      # Extract YES/NO from VLM response text         |   |
|   |      # Compare against ground truth boolean          |   |
|   |                                                      |   |
|   |  def evaluate_recorded_outputs(                      |   |
|   |      recorded: RecordedOutputs,                      |   |
|   |      expected_targets: Dict                          |   |
|   |  ) -> EvaluationReport:                              |   |
|   |      # Evaluate all recorded outputs                 |   |
|   |      # Aggregate into overall score                  |   |
|   +------------------------------------------------------+   |
|                                                              |
|   Evaluation Results (with Partial Credit):                  |
|   +------------------------------------------------------+   |
|   |  @dataclass PointingResult:                          |   |
|   |      target_object: str                              |   |
|   |      vlm_point: (x, y)                               |   |
|   |      gt_point: (x, y)                                |   |
|   |      distance_error: float  # pixels                 |   |
|   |      correct: bool          # inside target bbox     |   |
|   |      hit_same_category: bool # hit another of type   |   |
|   |      hit_object: str        # which object was hit   |   |
|   |      score: 1.0 (exact) | 0.5 (category) | 0.0       |   |
|   |                                                      |   |
|   |  @dataclass PathResult:                              |   |
|   |      start_correct: bool                             |   |
|   |      end_correct: bool                               |   |
|   |      path_valid: bool       # lenient collision      |   |
|   |      efficiency: float      # optimal/actual length  |   |
|   |      collisions: List[str]  # objects hit            |   |
|   |      direction: DirectionResult  # movement direction|   |
|   |                                                      |   |
|   |  @dataclass YesNoResult:                             |   |
|   |      answer_found: bool                              |   |
|   |      vlm_answer: bool                                |   |
|   |      gt_answer: bool                                 |   |
|   |      correct: bool                                   |   |
|   |      score: float           # 1.0 if correct         |   |
|   +------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
```

#### Pointing Evaluation: Partial Credit System

```
+--------------------------------------------------------------+
|              PARTIAL CREDIT FOR POINTING                      |
+--------------------------------------------------------------+
|                                                              |
|   Scoring Logic:                                             |
|   +------------------------------------------------------+   |
|   |  1. Check if point is inside TARGET's bounding box   |   |
|   |     → If YES: score = 1.0 (exact hit)                |   |
|   |                                                      |   |
|   |  2. Check if point is inside ANY same-category bbox  |   |
|   |     (e.g., pointed to stool_1 instead of stool_0)    |   |
|   |     → If YES: score = 0.5 (category hit)             |   |
|   |                                                      |   |
|   |  3. Otherwise:                                       |   |
|   |     → score = 0.0 (miss)                             |   |
|   +------------------------------------------------------+   |
|                                                              |
|   Example:                                                   |
|   +------------------------------------------------------+   |
|   |  Task: "Point to stool_0"                            |   |
|   |                                                      |   |
|   |  Case A: VLM points inside stool_0 bbox              |   |
|   |    → ✓ Exact hit: score = 1.0                        |   |
|   |                                                      |   |
|   |  Case B: VLM points inside stool_1 bbox              |   |
|   |    → ◐ Category hit: score = 0.5                     |   |
|   |       "hit stool_1 instead"                          |   |
|   |                                                      |   |
|   |  Case C: VLM points at lamp_0                        |   |
|   |    → ✗ Miss: score = 0.0                             |   |
|   +------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
```

#### Path Validation: Lenient Collision Detection

```
+--------------------------------------------------------------+
|              LENIENT COLLISION DETECTION                      |
+--------------------------------------------------------------+
|                                                              |
|   Problem: VLM draws floor-level walking paths, but object   |
|   bounding boxes include vertical extent (table legs, etc.)  |
|                                                              |
|   Solution: Shrink bboxes by 30% for collision checking      |
|   +------------------------------------------------------+   |
|   |                                                      |   |
|   |   Original bbox:        Shrunk bbox (30%):           |   |
|   |   ┌─────────────┐       ┌─────────────┐             |   |
|   |   │             │       │   ┌─────┐   │             |   |
|   |   │             │  -->  │   │     │   │             |   |
|   |   │             │       │   └─────┘   │             |   |
|   |   └─────────────┘       └─────────────┘             |   |
|   |                                                      |   |
|   |   Path passing through outer 30% = OK (floor space)  |   |
|   |   Path passing through inner core = COLLISION        |   |
|   +------------------------------------------------------+   |
|                                                              |
|   Implementation:                                            |
|   +------------------------------------------------------+   |
|   |  def _shrink_bbox(bbox, shrink_factor=0.3):          |   |
|   |      # Shrink by 30% on each side                    |   |
|   |      dx = (x2 - x1) * shrink_factor / 2              |   |
|   |      dy = (y2 - y1) * shrink_factor / 2              |   |
|   |      return [x1+dx, y1+dy, x2-dx, y2-dy]             |   |
|   |                                                      |   |
|   |  def is_path_valid(path, lenient_mode=True):         |   |
|   |      for segment in path:                            |   |
|   |          for obj in objects:                         |   |
|   |              bbox = _shrink_bbox(obj.bbox)           |   |
|   |              if line_intersects_bbox(segment, bbox): |   |
|   |                  return False, [obj.name]            |   |
|   |      return True, []                                 |   |
|   +------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
```

### 2.5 Component: `tasks.py`

**Purpose:** Define spatial intelligence evaluation tasks.

```
+--------------------------------------------------------------+
|                   Task Definitions                            |
+--------------------------------------------------------------+
|                                                              |
|   EGOCENTRIC vs ALLOCENTRIC:                                 |
|   +------------------------------------------------------+   |
|   |  EGOCENTRIC = Camera/Image viewpoint                 |   |
|   |    - The image IS what the model sees                |   |
|   |    - No viewpoint marker needed                      |   |
|   |    - Prompt: "You are currently viewing this scene"  |   |
|   |                                                      |   |
|   |  ALLOCENTRIC = Hypothetical marked viewpoint         |   |
|   |    - Image annotated with viewpoint marker           |   |
|   |    - Marker shows: position + facing direction       |   |
|   |    - Prompt: "Imagine you are standing at the        |   |
|   |              marked position (X) with the arrow      |   |
|   |              orientation."                           |   |
|   +------------------------------------------------------+   |
|                                                              |
|   Task Types (5 categories):                                 |
|   +------------------------------------------------------+   |
|   |  1. OBJECT_LOCALIZATION (Egocentric)                 |   |
|   |     - "Point to {target}"                            |   |
|   |     - No viewpoint marker (camera = viewpoint)       |   |
|   |     - Evaluates: pointing accuracy                   |   |
|   |                                                      |   |
|   |  2. EGOCENTRIC_PATH_PLANNING                         |   |
|   |     - Image has START marker at bottom-middle        |   |
|   |     - "Draw path from START to {target}"             |   |
|   |     - NO direction hint! (model locates object)      |   |
|   |     - Evaluates: start, end, validity, efficiency    |   |
|   |                                                      |   |
|   |  3. ALLOCENTRIC_PATH_PLANNING                        |   |
|   |     - Image has viewpoint marker + arrow direction   |   |
|   |     - "Imagine you are at (X) with arrow. Draw       |   |
|   |       path from A to B."                             |   |
|   |     - NO obstacle list (model sees image)            |   |
|   |     - Evaluates: start, end, validity, efficiency    |   |
|   |                                                      |   |
|   |  4. EGOCENTRIC_SPATIAL_QA                            |   |
|   |     - "Is the {target} on your left?" (YES/NO)       |   |
|   |     - Direction IS the question (tests spatial IQ)   |   |
|   |     - No viewpoint marker (camera = viewpoint)       |   |
|   |     - Evaluates: YES/NO answer + pointing accuracy   |   |
|   |                                                      |   |
|   |  5. ALLOCENTRIC_SPATIAL_QA                           |   |
|   |     - Image has viewpoint marker + facing direction  |   |
|   |     - "From marked viewpoint, is B left of A?"       |   |
|   |     - Ground truth computed in viewpoint's frame     |   |
|   |     - Evaluates: YES/NO answer + pointing accuracy   |   |
|   +------------------------------------------------------+   |
|                                                              |
|   @dataclass SpatialTask:                                    |
|   +------------------------------------------------------+   |
|   |  task_id: str                                        |   |
|   |  task_type: str                                      |   |
|   |  instruction: str      # Short description           |   |
|   |  prompt: str           # Full prompt for VLM         |   |
|   |  expected_targets: Dict # Ground truth info          |   |
|   |  annotated_image: Optional[PIL.Image]  # For alloc.  |   |
|   +------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
```

#### Viewpoint Annotation for Allocentric Tasks

Allocentric tasks require annotating the image with a viewpoint marker showing position and facing direction:

```
+--------------------------------------------------------------+
|              VIEWPOINT ANNOTATION SYSTEM                      |
+--------------------------------------------------------------+
|                                                              |
|   ViewpointAnnotator (spatial_agent/viewpoint_annotator.py)  |
|   +------------------------------------------------------+   |
|   |  def add_viewpoint_marker(position, facing, label):  |   |
|   |      # position: (x, y) in image coordinates         |   |
|   |      # facing: "camera", "away", "left", "right"     |   |
|   |      # label: "YOU ARE HERE"                         |   |
|   |                                                      |   |
|   |  Visual elements:                                    |   |
|   |  - Blue circle at position                           |   |
|   |  - Red arrow showing facing direction                |   |
|   |  - Text label above marker                           |   |
|   +------------------------------------------------------+   |
|                                                              |
|   Example annotated image:                                   |
|   +------------------------------------------------------+   |
|   |                                                      |   |
|   |     [scene with objects]                             |   |
|   |                                                      |   |
|   |              YOU ARE HERE                            |   |
|   |                  [●]→                                |   |
|   |              facing: right                           |   |
|   |                                                      |   |
|   +------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
```

#### Ground Truth Computation for Allocentric Tasks

Allocentric spatial relations are computed relative to the viewpoint's reference frame:

```
+--------------------------------------------------------------+
|           ALLOCENTRIC GROUND TRUTH COMPUTATION                |
+--------------------------------------------------------------+
|                                                              |
|   Input:                                                     |
|   - viewpoint_position: (vx, vy)                            |
|   - facing: "camera" | "away" | "left" | "right"            |
|   - reference_object: position (rx, ry)                     |
|   - target_object: position (tx, ty)                        |
|   - relation: "left" | "right" | "front" | "behind"         |
|                                                              |
|   Step 1: Transform to viewpoint-centered coordinates        |
|   +------------------------------------------------------+   |
|   |  # Relative positions from viewpoint                 |   |
|   |  ref_dx = rx - vx                                    |   |
|   |  ref_dy = ry - vy                                    |   |
|   |  target_dx = tx - vx                                 |   |
|   |  target_dy = ty - vy                                 |   |
|   +------------------------------------------------------+   |
|                                                              |
|   Step 2: Rotate based on facing direction                   |
|   +------------------------------------------------------+   |
|   |  if facing == "camera":                              |   |
|   |      # Facing up (toward top of image)               |   |
|   |      # No rotation needed                            |   |
|   |  elif facing == "away":                              |   |
|   |      # Facing down - 180° rotation                   |   |
|   |      ref_dx, ref_dy = -ref_dx, -ref_dy              |   |
|   |      target_dx, target_dy = -target_dx, -target_dy  |   |
|   |  elif facing == "left":                              |   |
|   |      # Facing left - 90° CCW rotation               |   |
|   |      ref_dx, ref_dy = ref_dy, -ref_dx               |   |
|   |      target_dx, target_dy = target_dy, -target_dx   |   |
|   |  elif facing == "right":                             |   |
|   |      # Facing right - 90° CW rotation               |   |
|   |      ref_dx, ref_dy = -ref_dy, ref_dx               |   |
|   |      target_dx, target_dy = -target_dy, target_dx   |   |
|   +------------------------------------------------------+   |
|                                                              |
|   Step 3: Evaluate relation in rotated frame                 |
|   +------------------------------------------------------+   |
|   |  threshold = image_width * 0.05  # 5% tolerance     |   |
|   |                                                      |   |
|   |  if relation == "left":                              |   |
|   |      return target_dx < ref_dx - threshold          |   |
|   |  elif relation == "right":                           |   |
|   |      return target_dx > ref_dx + threshold          |   |
|   |  elif relation == "front":                           |   |
|   |      return target_dy < ref_dy - threshold          |   |
|   |  elif relation == "behind":                          |   |
|   |      return target_dy > ref_dy + threshold          |   |
|   +------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
```

#### Task Generation: Simplified Prompts

**Design Principle:** Prompts are minimal - no obstacle lists, no coordinate hints. Model reasons from image alone.

**Egocentric Path Planning:**
```
+--------------------------------------------------------------+
|   [Image has START marker at bottom-middle]                   |
|                                                               |
|   You are currently viewing this scene. Your current position |
|   is marked as "START HERE".                                  |
|                                                               |
|   TASK: Draw a walking path from your position to the         |
|         leftmost lamp. Make sure to avoid obstacles.          |
|                                                               |
|   Use draw_path with waypoint coordinates to show the route.  |
+--------------------------------------------------------------+
```

**Allocentric Path Planning:**
```
+--------------------------------------------------------------+
|   [Image has viewpoint marker with arrow showing direction]   |
|                                                               |
|   Imagine you are standing at the marked position (X) with    |
|   the arrow orientation.                                      |
|                                                               |
|   TASK: Draw a walking path from the table to the lamp.       |
|         Make sure to avoid obstacles.                         |
|                                                               |
|   Use draw_path with waypoint coordinates to show the route.  |
+--------------------------------------------------------------+
```

**Egocentric Spatial QA:**
```
+--------------------------------------------------------------+
|   You are currently viewing this scene.                       |
|                                                               |
|   QUESTION: Is the lamp on your left?                         |
|                                                               |
|   Answer YES or NO, then use point_to to indicate the         |
|   lamp's location.                                            |
+--------------------------------------------------------------+
```

**Allocentric Spatial QA:**
```
+--------------------------------------------------------------+
|   [Image has viewpoint marker with arrow showing direction]   |
|                                                               |
|   Imagine you are standing at the marked position (X) with    |
|   the arrow orientation.                                      |
|                                                               |
|   QUESTION: From your viewpoint, is the stool to the left     |
|             of the table?                                     |
|                                                               |
|   Answer YES or NO, then use point_to to indicate the         |
|   stool's location.                                           |
+--------------------------------------------------------------+
```

#### Critical: Facing Direction Changes Ground Truth

**Same question, different answers based on viewpoint orientation:**

```
+--------------------------------------------------------------+
|           ALLOCENTRIC GROUND TRUTH EXAMPLE                    |
+--------------------------------------------------------------+
|                                                               |
|   Question: "Is the stool to the LEFT of the table?"          |
|                                                               |
|   Object positions in image:                                  |
|     stool.x = 3552, table.x = 2832                           |
|     In image coords: stool is to the RIGHT of table          |
|                                                               |
|   Facing CAMERA (↑):  Answer = NO   (standard view)          |
|   Facing AWAY (↓):    Answer = YES  (180° flip)              |
|   Facing LEFT (←):    Answer = NO   (different axis)         |
|                                                               |
|   This tests TRUE allocentric reasoning!                      |
+--------------------------------------------------------------+
```

#### Allocentric QA: Category Filtering

To avoid self-referential questions, allocentric QA tasks ensure different object categories:

```
+--------------------------------------------------------------+
|              ALLOCENTRIC QA CATEGORY FILTERING                |
+--------------------------------------------------------------+
|                                                              |
|   Problem: "From the table's position, is the table to the   |
|            left?" is a nonsensical question.                 |
|                                                              |
|   Solution: Ensure reference and target are different types  |
|   +------------------------------------------------------+   |
|   |  # Group objects by category                         |   |
|   |  by_category = {                                     |   |
|   |      "stool": [stool_0, stool_1, stool_2],          |   |
|   |      "lamp": [lamp_0, lamp_1, lamp_2],              |   |
|   |      "table": [table_0],                            |   |
|   |      "person": [person]                             |   |
|   |  }                                                   |   |
|   |                                                      |   |
|   |  # Select from DIFFERENT categories                  |   |
|   |  reference = random.choice(by_category["table"])     |   |
|   |  target = random.choice(by_category["lamp"])         |   |
|   +------------------------------------------------------+   |
|                                                              |
|   Valid question: "From the table's position, is the lamp    |
|                    to the left?"                             |
|                                                              |
+--------------------------------------------------------------+
```

### 2.6 Component: `embodied_agent.py`

**Purpose:** Run VLM with primitive tools and evaluate outputs.

```
+--------------------------------------------------------------+
|                   EmbodiedSpatialAgent                        |
+--------------------------------------------------------------+
|                                                              |
|   class EmbodiedSpatialAgent:                                |
|   +------------------------------------------------------+   |
|   |  def __init__(                                       |   |
|   |      model_name="claude-sonnet-4-20250514",          |   |
|   |      temperature=0.0                                 |   |
|   |  ):                                                  |   |
|   |      # Initialize LLM                                |   |
|   |      # Will setup tools per-task                     |   |
|   |                                                      |   |
|   |  def run_task(                                       |   |
|   |      task: SpatialTask,                              |   |
|   |      image_path: str                                 |   |
|   |  ) -> Dict:                                          |   |
|   |      # 1. Setup fresh tools (RecordedOutputs)        |   |
|   |      # 2. Encode image as base64                     |   |
|   |      # 3. Create message with image + task prompt    |   |
|   |      # 4. Run agent (LangGraph ReAct)                |   |
|   |      # 5. Return recorded outputs                    |   |
|   |                                                      |   |
|   |  def run_and_evaluate(                               |   |
|   |      task: SpatialTask,                              |   |
|   |      image_path: str,                                |   |
|   |      ground_truth: GroundTruth                       |   |
|   |  ) -> EvaluationReport:                              |   |
|   |      # 1. Run task                                   |   |
|   |      # 2. Evaluate recorded outputs against GT       |   |
|   |      # 3. Return evaluation report                   |   |
|   +------------------------------------------------------+   |
|                                                              |
|   class EmbodiedEvaluationRunner:                            |
|   +------------------------------------------------------+   |
|   |  def __init__(                                       |   |
|   |      spatial_graph_path: str,                        |   |
|   |      image_path: str,                                |   |
|   |      model_name: str                                 |   |
|   |  ):                                                  |   |
|   |      # Load ground truth                             |   |
|   |      # Initialize agent                              |   |
|   |                                                      |   |
|   |  def run_task_suite(                                 |   |
|   |      tasks: List[SpatialTask]                        |   |
|   |  ) -> Dict:                                          |   |
|   |      # Run all tasks                                 |   |
|   |      # Aggregate results by task type                |   |
|   |      # Compute overall score                         |   |
|   |                                                      |   |
|   |  def print_summary(results: Dict):                   |   |
|   |      # Pretty print evaluation summary               |   |
|   +------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
```

### 2.7 System Prompt for VLM

```
+--------------------------------------------------------------+
|                        SYSTEM PROMPT                          |
+--------------------------------------------------------------+
|                                                              |
|  You are an embodied spatial reasoning agent analyzing       |
|  a scene.                                                    |
|                                                              |
|  ## Your Capabilities                                        |
|                                                              |
|  You have access to ALL of the following tools to express    |
|  your spatial understanding. Use whichever tools best help   |
|  you complete the task:                                      |
|                                                              |
|  1. point_to(x, y, label) - Point to a location              |
|     - Use when asked "where is X?" or "point to X"           |
|     - x: horizontal position (0 = left edge)                 |
|     - y: vertical position (0 = top edge)                    |
|                                                              |
|  2. draw_path(points, label) - Draw a path                   |
|     - Use when asked to navigate or plan a route             |
|     - points: list of (x, y) coordinates                     |
|     - First point = start, last point = destination          |
|                                                              |
|  3. mark_region(x1, y1, x2, y2, label) - Mark region         |
|     - Use to highlight an object or area                     |
|     - Provide top-left and bottom-right corners              |
|                                                              |
|  4. move_to(x, y) - Move to position                         |
|     - Use for step-by-step navigation                        |
|                                                              |
|  5. rotate(angle) - Rotate facing direction                  |
|     - Positive = clockwise, Negative = counter-clockwise     |
|                                                              |
|  6. look_at(x, y) - Turn to face location                    |
|                                                              |
|  ## Image Coordinate System                                  |
|                                                              |
|  - Origin (0, 0) is at the TOP-LEFT corner                   |
|  - X increases to the RIGHT                                  |
|  - Y increases DOWNWARD                                      |
|                                                              |
|  ## Task Types                                               |
|                                                              |
|  EGOCENTRIC tasks: The image IS your viewpoint.              |
|  - Prompt says "You are currently viewing this scene"        |
|  - Left/right/front/back are from the camera's perspective   |
|                                                              |
|  ALLOCENTRIC tasks: A marker shows a hypothetical viewpoint. |
|  - Image has "YOU ARE HERE" marker with facing arrow         |
|  - Mentally adopt that position and facing direction         |
|  - Spatial relations are from the marker's perspective       |
|                                                              |
|  IMPORTANT:                                                  |
|  - You must LOOK at the image to determine positions         |
|  - Estimate coordinates based on what you SEE                |
|  - Use tools to express your spatial reasoning               |
|  - You may use MULTIPLE tools if helpful for the task        |
|                                                              |
+--------------------------------------------------------------+
```

### 2.8 Execution Flow

```
+--------------------------------------------------------------+
|                    Task Execution Flow                        |
+--------------------------------------------------------------+
|                                                              |
|   1. SETUP                                                   |
|   +------------------------------------------------------+   |
|   |  - Load task (prompt, expected targets)              |   |
|   |  - Initialize fresh RecordedOutputs                  |   |
|   |  - Initialize EmbodiedState                          |   |
|   |  - Create tools bound to outputs/state               |   |
|   |  - Create LangGraph ReAct agent                      |   |
|   +------------------------------------------------------+   |
|                           |                                  |
|                           v                                  |
|   2. VLM INPUT                                               |
|   +------------------------------------------------------+   |
|   |  [Image: scene.png (base64 encoded)]                 |   |
|   |                                                      |   |
|   |  Task: Point to the table in this scene.             |   |
|   |                                                      |   |
|   |  Image dimensions: 800w x 600h pixels                |   |
|   +------------------------------------------------------+   |
|                           |                                  |
|                           v                                  |
|   3. VLM REASONING + TOOL CALLS                              |
|   +------------------------------------------------------+   |
|   |  VLM: "Looking at the image, I can see a wooden      |   |
|   |        table in the center-right portion of the      |   |
|   |        scene. It appears to be around x=450, y=320.  |   |
|   |        I'll use point_to to indicate its location."  |   |
|   |                                                      |   |
|   |  Tool call: point_to(450, 320, "table")              |   |
|   +------------------------------------------------------+   |
|                           |                                  |
|                           v                                  |
|   4. OUTPUTS RECORDED                                        |
|   +------------------------------------------------------+   |
|   |  RecordedOutputs.points = [                          |   |
|   |      {                                               |   |
|   |          "x": 450,                                   |   |
|   |          "y": 320,                                   |   |
|   |          "label": "table"                            |   |
|   |      }                                               |   |
|   |  ]                                                   |   |
|   +------------------------------------------------------+   |
|                           |                                  |
|                           v                                  |
|   5. EVALUATION                                              |
|   +------------------------------------------------------+   |
|   |  Ground Truth:                                       |   |
|   |      table.bbox_center = (455, 315)                  |   |
|   |                                                      |   |
|   |  VLM Output:                                         |   |
|   |      pointed_position = (450, 320)                   |   |
|   |                                                      |   |
|   |  Distance Error:                                     |   |
|   |      sqrt((455-450)^2 + (315-320)^2) = 7.07 pixels   |   |
|   |                                                      |   |
|   |  Score: 1.0 - (7.07 / 50) = 0.86                     |   |
|   |  (threshold = 50px for "correct")                    |   |
|   +------------------------------------------------------+   |
|                           |                                  |
|                           v                                  |
|   6. RESULT                                                  |
|   +------------------------------------------------------+   |
|   |  EvaluationReport:                                   |   |
|   |      overall_score: 0.86                             |   |
|   |      pointing_results: [                             |   |
|   |          PointingResult(                             |   |
|   |              target="table",                         |   |
|   |              distance_error=7.07,                    |   |
|   |              correct=True                            |   |
|   |          )                                           |   |
|   |      ]                                               |   |
|   +------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
```

---

## Stage 3: Allocentric Query Evaluation

### 3.1 Query Generation Pipeline

```
+--------------------------------------------------------------+
|                   Query Generation Flow                       |
+--------------------------------------------------------------+
|                                                              |
|   spatial_graph.json                                         |
|           |                                                  |
|           v                                                  |
|   +------------------------------------------------------+   |
|   |  generate_queries.py                                 |   |
|   |                                                      |   |
|   |  1. Load spatial graph (object positions, z-order)   |   |
|   |                                                      |   |
|   |  2. Select anchor entity (e.g., "table")             |   |
|   |                                                      |   |
|   |  3. Estimate anchor orientation using GPT-4o         |   |
|   |     - Send image + prompt to GPT-4o                  |   |
|   |     - "Which direction is the table facing?"         |   |
|   |     - Response: "facing_camera" / "facing_left" etc  |   |
|   |                                                      |   |
|   |  4. Generate queries based on relationships:         |   |
|   |     - left_right: "Is lamp left or right of table?"  |   |
|   |     - front_behind: "Is chair in front of table?"    |   |
|   |     - nearest_object: "What's closest to table?"     |   |
|   |     - etc.                                           |   |
|   |                                                      |   |
|   |  5. Compute ground truth answers from spatial graph  |   |
|   +------------------------------------------------------+   |
|           |                                                  |
|           v                                                  |
|   queries.json                                               |
|   +------------------------------------------------------+   |
|   |  {                                                   |   |
|   |    "anchor_entity": {"name": "table"},               |   |
|   |    "anchor_orientation": {                           |   |
|   |      "orientation": "facing_camera",                 |   |
|   |      "confidence": 0.85                              |   |
|   |    },                                                |   |
|   |    "queries": [                                      |   |
|   |      {                                               |   |
|   |        "query_id": "q_001",                          |   |
|   |        "query_type": "left_right",                   |   |
|   |        "prompt": "Is the lamp to the left or right   |   |
|   |                   of the table?",                    |   |
|   |        "answer": "left",                             |   |
|   |        "targets": ["lamp_0"]                         |   |
|   |      },                                              |   |
|   |      ...                                             |   |
|   |    ]                                                 |   |
|   |  }                                                   |   |
|   +------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
```

### 3.2 Query Integration

```
+--------------------------------------------------------------+
|                   query_integration.py                        |
+--------------------------------------------------------------+
|                                                              |
|   class QueryLoader:                                         |
|   +------------------------------------------------------+   |
|   |  def __init__(queries_json_path: str):               |   |
|   |      # Load queries from generate_queries.py output  |   |
|   |      # Parse anchor entity and orientation           |   |
|   |      # Parse all queries                             |   |
|   |                                                      |   |
|   |  def to_spatial_tasks() -> List[SpatialTask]:        |   |
|   |      # Convert queries to SpatialTask format         |   |
|   |      # Map query types to task types                 |   |
|   |      # Build prompts with tool instructions          |   |
|   +------------------------------------------------------+   |
|                                                              |
|   Query Type -> Task Type Mapping:                           |
|   +------------------------------------------------------+   |
|   |  left_right     -> SPATIAL_RELATION_QA               |   |
|   |  front_behind   -> SPATIAL_RELATION_QA               |   |
|   |  above_below    -> SPATIAL_RELATION_QA               |   |
|   |  visibility     -> SPATIAL_RELATION_QA               |   |
|   |  nearest_object -> SPATIAL_RELATION_QA               |   |
|   |  farthest_object -> SPATIAL_RELATION_QA              |   |
|   |  closest_left   -> SPATIAL_RELATION_QA               |   |
|   |  closest_right  -> SPATIAL_RELATION_QA               |   |
|   +------------------------------------------------------+   |
|                                                              |
|   Evaluation Flow:                                           |
|   +------------------------------------------------------+   |
|   |  queries.json                                        |   |
|   |        |                                             |   |
|   |        v                                             |   |
|   |  QueryLoader.to_spatial_tasks()                      |   |
|   |        |                                             |   |
|   |        v                                             |   |
|   |  List[SpatialTask]                                   |   |
|   |        |                                             |   |
|   |        v                                             |   |
|   |  EmbodiedEvaluationRunner.run_task_suite()           |   |
|   |        |                                             |   |
|   |        v                                             |   |
|   |  Evaluation Results                                  |   |
|   +------------------------------------------------------+   |
|                                                              |
+--------------------------------------------------------------+
```

### 3.3 Unified Allocentric Evaluation (`run_allocentric_eval.py`)

```
+--------------------------------------------------------------+
|                   Full Pipeline (--mode full)                 |
+--------------------------------------------------------------+
|                                                              |
|   Step 1: Generate Queries                                   |
|   +------------------------------------------------------+   |
|   |  python generate_queries.py                          |   |
|   |      --input spatial_graph.json                      |   |
|   |      --output queries.json                           |   |
|   |      --anchor "table"                                |   |
|   |                                                      |   |
|   |  Uses GPT-4o for orientation estimation              |   |
|   +------------------------------------------------------+   |
|                           |                                  |
|                           v                                  |
|   Step 2: Load and Convert Queries                           |
|   +------------------------------------------------------+   |
|   |  tasks, gt, loader = load_and_convert_queries(       |   |
|   |      queries_path,                                   |   |
|   |      spatial_graph_path                              |   |
|   |  )                                                   |   |
|   +------------------------------------------------------+   |
|                           |                                  |
|                           v                                  |
|   Step 3: Run Evaluation                                     |
|   +------------------------------------------------------+   |
|   |  runner = EmbodiedEvaluationRunner(                  |   |
|   |      spatial_graph_path,                             |   |
|   |      image_path,                                     |   |
|   |      model_name="claude-sonnet-4-20250514"           |   |
|   |  )                                                   |   |
|   |                                                      |   |
|   |  results = runner.run_task_suite(tasks)              |   |
|   +------------------------------------------------------+   |
|                           |                                  |
|                           v                                  |
|   Step 4: Analyze by Query Type                              |
|   +------------------------------------------------------+   |
|   |  by_query_type = {                                   |   |
|   |      'left_right': {                                 |   |
|   |          'count': 5,                                 |   |
|   |          'accuracy': 0.80,                           |   |
|   |          'avg_score': 0.75                           |   |
|   |      },                                              |   |
|   |      'front_behind': {                               |   |
|   |          'count': 4,                                 |   |
|   |          'accuracy': 0.50,                           |   |
|   |          'avg_score': 0.60                           |   |
|   |      },                                              |   |
|   |      ...                                             |   |
|   |  }                                                   |   |
|   +------------------------------------------------------+   |
|                           |                                  |
|                           v                                  |
|   Output: allocentric_eval_results.json                      |
|                                                              |
+--------------------------------------------------------------+
```

---

## Complete End-to-End Example

### Option A: Manual Prompts (No API for Ground Truth)

```bash
# Step 1: Generate ground truth from image (local models only)
python depth_sam3_connector.py \
    --image scene.jpg \
    --prompts "table" "chair" "person" "lamp" \
    --output_dir spatial_outputs/

# Step 2: Run embodied spatial intelligence evaluation
ANTHROPIC_API_KEY=xxx python run_embodied_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --task suite \
    --num-tasks 20
```

### Option B: Unified Pipeline (Auto-Detect Objects)

```bash
# Step 1: Auto-detect objects + prepare queries (GPT-4o)
OPENAI_API_KEY=xxx python generate_queries.py \
    --image scene.jpg \
    --auto-detect-objects \
    --anchor "table" \
    -o query_outputs/

# Outputs:
#   query_outputs/prompts.json   <- For SAM3
#   query_outputs/queries.json   <- Preliminary (or full if --input provided)

# Step 2: Generate ground truth using detected prompts (local)
python depth_sam3_connector.py \
    --image scene.jpg \
    --prompts-file query_outputs/prompts.json \
    --output_dir spatial_outputs/

# Step 3: Generate full queries from spatial graph (GPT-4o for orientation)
OPENAI_API_KEY=xxx python generate_queries.py \
    --input spatial_outputs/scene_spatial_graph.json \
    --anchor "table" \
    -o query_outputs/

# Step 4: Run evaluation (Claude or Qwen3-VL)
ANTHROPIC_API_KEY=xxx python run_embodied_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --task suite

# Or with local Qwen3-VL (no API key needed)
python run_embodied_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --provider vllm \
    --task suite
```

---

## Unified Preparation Pipeline

The `generate_queries.py` script now supports a unified preparation flow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED PREPARATION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MODE 1: From Image (--image --auto-detect-objects)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Image                                                              │    │
│  │    │                                                                │    │
│  │    ▼                                                                │    │
│  │  GPT-4o: "List all objects in this image"                          │    │
│  │    │                                                                │    │
│  │    ▼                                                                │    │
│  │  prompts.json ─────────────────────► depth_sam3_connector.py       │    │
│  │  ["table", "chair", "lamp", ...]         │                         │    │
│  │                                          ▼                         │    │
│  │                                   spatial_graph.json               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  MODE 2: From Graph (--input)                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  spatial_graph.json                                                 │    │
│  │    │                                                                │    │
│  │    ▼                                                                │    │
│  │  GPT-4o: Estimate anchor orientation                               │    │
│  │    │                                                                │    │
│  │    ▼                                                                │    │
│  │  queries.json ─────────────────────► run_embodied_eval.py          │    │
│  │  (spatial reasoning queries)             │                         │    │
│  │                                          ▼                         │    │
│  │                                   Evaluation Results               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CLI Reference

```bash
# Mode 1: Auto-detect objects from image
python generate_queries.py \
    --image scene.jpg \
    --auto-detect-objects \
    --anchor "table" \           # Optional: specify anchor
    --additional-prompts "book"  # Optional: add more objects
    -o outputs/

# Mode 2: Generate queries from spatial graph
python generate_queries.py \
    --input spatial_graph.json \
    --anchor "table" \           # Required: specify anchor
    -o outputs/

# depth_sam3_connector.py now supports prompts from file
python depth_sam3_connector.py \
    --image scene.jpg \
    --prompts-file outputs/prompts.json \  # From generate_queries.py
    --prompts "extra_object" \             # Optional: additional prompts
    -o spatial_outputs/
```

**Expected Output:**
```
============================================================
EVALUATION SUMMARY
============================================================

Overall Score: 72.3%
Total Tasks: 20

By Task Type:
  object_localization:
    Count: 4
    Average: 85.2%
  path_planning:
    Count: 4
    Average: 68.0%
  spatial_relation_qa:
    Count: 8
    Average: 70.0%
  perspective_shift:
    Count: 4
    Average: 65.0%

============================================================
ANALYSIS BY QUERY TYPE
============================================================

  left_right:
    Count: 3
    Accuracy: 100.0%
    Avg Score: 0.95

  front_behind:
    Count: 2
    Accuracy: 50.0%
    Avg Score: 0.60

  nearest_object:
    Count: 3
    Accuracy: 66.7%
    Avg Score: 0.72

============================================================
```

---

## File Reference

```
cherry/
+-- depth_sam3_connector.py      # Stage 1: Perception pipeline -> Ground Truth
+-- run_sam3.py                  # SAM3 segmentation
+-- extract_depth.py             # Depth extraction
+-- spatial_graph.py             # Graph analysis
+-- run_embodied_eval.py         # Stage 2: Embodied evaluation runner
+-- run_allocentric_eval.py      # Stage 3: Allocentric Q&A evaluation
+-- generate_queries.py          # Query generation with GPT-4o
|
+-- spatial_agent/
|   +-- __init__.py              # Package exports
|   +-- annotator.py             # Image annotation
|   +-- state.py                 # Agent state management
|   +-- tools.py                 # Navigation tools
|   +-- agent.py                 # Navigation agents
|   |
|   +-- primitive_tools.py       # OUTPUT MECHANISM tools
|   +-- ground_truth.py          # Ground truth loader
|   +-- evaluator.py             # Compares outputs to ground truth
|   +-- tasks.py                 # Task definitions
|   +-- embodied_agent.py        # Embodied spatial agent
|   |
|   +-- allocentric.py           # Allocentric relationships
|   +-- allocentric_tools.py     # Allocentric LangChain tools
|   +-- allocentric_eval.py      # Allocentric evaluation
|   +-- query_integration.py     # Query to task conversion
|
+-- sam3/                        # Submodule: SAM3 model
+-- dinov3/                      # Submodule: DINOv3 depth
|
+-- requirements.txt             # Dependencies
+-- README.md                    # User documentation
+-- PIPELINE_FLOW.md             # This file
```

---

## Model Configuration

### VLM Providers for Evaluation

The evaluation framework supports multiple VLM backends via a provider abstraction:

| Provider | Model | Setup | Tool Calling |
|----------|-------|-------|--------------|
| `claude` | `claude-sonnet-4-20250514` | API key (ANTHROPIC_API_KEY) | Native |
| `vllm` | `Qwen/Qwen3-VL-8B-Instruct` | vLLM server on GPU | Hermes parser |
| `openai` | `gpt-4o` | API key (OPENAI_API_KEY) | Native |
| `ollama` | `llava` | Local Ollama server | Via prompting |

**Running with vLLM (recommended for open-source models):**
```bash
# Start vLLM server with tool calling
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 8000 \
    --gpu-memory-utilization 0.35

# Run evaluation
python run_embodied_eval.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --provider vllm \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --task suite
```

### Other Pipeline Models

| Component | Model | Purpose |
|-----------|-------|---------|
| Object Detection | GPT-4o | Auto-detect objects in image (optional) |
| Query Generation | GPT-4o | Anchor orientation estimation |
| Segmentation | SAM3 (facebook/sam3-hiera-large) | Object detection |
| Depth Estimation | DPT (Intel/dpt-large) | Monocular depth |

### API Key Requirements by Pipeline Mode

| Pipeline Mode | GPT-4o (OPENAI) | VLM Eval | Description |
|---------------|-----------------|----------|-------------|
| **Manual + Claude** | Not needed | `ANTHROPIC_API_KEY` | Manual prompts, Claude eval |
| **Manual + Qwen3-VL** | Not needed | Not needed (local) | Fully local pipeline |
| **Auto-detect + Claude** | `OPENAI_API_KEY` | `ANTHROPIC_API_KEY` | Auto object detection |
| **Auto-detect + Qwen3-VL** | `OPENAI_API_KEY` | Not needed (local) | Auto detection, local eval |

---

## Primitive Tools: Complete Reference

### Tool Availability

**Important:** The VLM has access to **ALL 6 primitive tools** for every task. The "expected tool" in task definitions indicates which tool we primarily evaluate against, but the model is free to use any combination of tools to solve the task.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ALL TOOLS AVAILABLE PER TASK                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Task: "Point to the table"                                                │
│                                                                              │
│   VLM has access to:                                                        │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                          │
│   │  point_to   │ │  draw_path  │ │ mark_region │  ← All available        │
│   │  (primary)  │ │             │ │             │                          │
│   └─────────────┘ └─────────────┘ └─────────────┘                          │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                          │
│   │   move_to   │ │   rotate    │ │   look_at   │  ← All available        │
│   └─────────────┘ └─────────────┘ └─────────────┘                          │
│                                                                              │
│   The model might:                                                          │
│   1. Use point_to(x, y, "table") - Primary expected                        │
│   2. Also use mark_region() to highlight the table area                    │
│   3. Use look_at() to indicate attention direction                         │
│                                                                              │
│   Evaluation focuses on the primary expected tool output.                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tool Definitions

| Tool | Signature | Purpose | Records |
|------|-----------|---------|---------|
| **point_to** | `point_to(x, y, label)` | Indicate a specific location | `{x, y, label}` |
| **draw_path** | `draw_path(points, label)` | Draw navigation route | `{points: [(x,y)...], start, end, length}` |
| **mark_region** | `mark_region(x1, y1, x2, y2, label)` | Highlight object/area | `{bbox, label, center, area}` |
| **move_to** | `move_to(x, y)` | Navigate to position | `{from, to, distance}` |
| **rotate** | `rotate(angle)` | Change facing direction | `angle` |
| **look_at** | `look_at(x, y)` | Direct gaze/attention | `{target, from_position, facing_angle}` |

### Task Types and Expected Tools

| Task Type | Example | Primary Tool | Why |
|-----------|---------|--------------|-----|
| **Object Localization** | "Point to the table" | `point_to` | Express location understanding |
| **Path Planning** | "Draw path from chair to lamp" | `draw_path` | Express navigation planning |
| **Egocentric Navigation** | "You're at (100,200). Navigate to table" | `draw_path` | Express route from viewpoint |
| **Allocentric Navigation** | "From the chair, draw path to lamp" | `draw_path` | Express route from reference |
| **Spatial Relation QA** | "Is lamp left of table? Point to it" | `point_to` | Verify understanding via location |
| **Perspective Shift** | "From table's view, what's on your left?" | `point_to` | Express perspective reasoning |

---

## Evaluation Metrics: Complete Reference

### Metric Definitions

| Metric | Formula | Threshold | Score |
|--------|---------|-----------|-------|
| **Pointing Accuracy** | `point_inside_bbox(vlm_point, target)` | Inside target bbox | 1.0 exact, 0.5 same-category, 0.0 miss |
| **Path Start** | `point_inside_bbox(path[0], start)` | Inside start bbox | 0.25 weight |
| **Path End** | `point_inside_bbox(path[-1], goal)` | Inside goal bbox | 0.25 weight |
| **Path Validity** | `no_collisions(path, shrunk_obstacles)` | 0 collisions (lenient) | 0.25 weight |
| **Path Efficiency** | `optimal_length / actual_length` | 1.0 = perfect | 0.25 × efficiency |
| **Yes/No QA** | `vlm_answer == gt_answer` | Exact match | 1.0 if match, else 0.0 |

### Pointing Score: Partial Credit

```
Pointing Score:
  1. Check if point inside TARGET's bounding box
     → YES: score = 1.0 (exact hit) ✓

  2. Check if point inside ANY same-category object's bbox
     → YES: score = 0.5 (category hit) ◐

  3. Otherwise:
     → score = 0.0 (miss) ✗

Example (target = stool_0):
  - VLM points at (3500, 3400), inside stool_0 bbox → 1.0 ✓
  - VLM points at (3200, 3100), inside stool_1 bbox → 0.5 ◐
  - VLM points at (4800, 2400), inside person bbox  → 0.0 ✗
```

### Path Score Breakdown

```
Path Score = 0.25 × start_correct
           + 0.25 × end_correct
           + 0.25 × path_valid (lenient collision)
           + 0.25 × min(efficiency, 1.0)

Lenient Collision Detection:
  - Bboxes shrunk by 30% for floor-level path realism
  - Allows paths near object edges (where floor space exists)
  - Only collides with core 70% of object footprint

Example:
  - Start inside stool_0 bbox: ✓ (0.25)
  - End inside person bbox: ✓ (0.25)
  - No collisions (after 30% shrink): ✓ (0.25)
  - Path length 1500px, optimal 1200px: efficiency = 0.8 (0.20)
  - Total: 0.95
```

### Spatial QA Score

```
Spatial QA Score = 0.5 × yes_no_correct + 0.5 × pointing_score

Yes/No Extraction:
  - Scans VLM response for "YES" or "NO" (case-insensitive)
  - Compares against ground truth boolean

Example (task: "Is the person in the foreground?"):
  - VLM response: "NO. The person is in the background..."
  - VLM answer: NO (correct) → 0.5
  - VLM points at (4800, 2400), inside person bbox → 0.5
  - Total: 1.0
```

### Ground Truth Source

All ground truth comes from the SAM3 + Depth perception pipeline:

```
spatial_graph.json
├── nodes[].bbox_center     → Object position for pointing evaluation
├── nodes[].bbox            → Object bounds for bbox hit detection & collision
├── nodes[].z_order         → Depth ordering for front/behind relations
│                             (HIGHER z_order = closer to camera/foreground)
└── nodes[].relative_depth  → Continuous depth for proximity calculations

Z-Order Interpretation:
├── z_order = 0  → Background (farthest from camera)
├── z_order = N  → Foreground (closest to camera)
└── Higher z = visually in front

Spatial Relations (computed from above):
├── left_of(a, b)     → a.center_x < b.center_x - threshold
├── right_of(a, b)    → a.center_x > b.center_x + threshold
├── above(a, b)       → a.center_y < b.center_y - threshold
├── below(a, b)       → a.center_y > b.center_y + threshold
├── in_front_of(a, b) → a.z_order > b.z_order (HIGHER z = in front)
├── behind(a, b)      → a.z_order < b.z_order (LOWER z = behind)
└── near(a, b)        → distance < 15% of image diagonal
```
