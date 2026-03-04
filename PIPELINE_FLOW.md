# Cherry Pipeline Flow

This document provides a comprehensive end-to-end walkthrough of the Cherry pipeline, from raw image input to agentic spatial reasoning evaluation.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CHERRY PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 1: PERCEPTION                                                         │
│  ┌──────────────┐    ┌──────────────┐                                       │
│  │    Image     │    │    Text      │                                       │
│  │   (scene)    │    │   Prompts    │                                       │
│  └──────┬───────┘    └──────┬───────┘                                       │
│         │                   │                                                │
│         └─────────┬─────────┘                                                │
│                   ▼                                                          │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                  depth_sam3_connector.py                        │         │
│  │  ┌─────────────────────┐    ┌─────────────────────┐            │         │
│  │  │      SAM3           │    │   Depth Extractor   │            │         │
│  │  │   (Segmentation)    │    │   (DPT / DINOv3)    │            │         │
│  │  │                     │    │                     │            │         │
│  │  │  Prompt → Masks     │    │  Image → Depth Map  │            │         │
│  │  │  + Bounding Boxes   │    │  + Statistics       │            │         │
│  │  └──────────┬──────────┘    └──────────┬──────────┘            │         │
│  │             │                          │                        │         │
│  │             └────────────┬─────────────┘                        │         │
│  │                          ▼                                      │         │
│  │             ┌─────────────────────────┐                         │         │
│  │             │     Z-Order Ranking     │                         │         │
│  │             │  (median depth → rank)  │                         │         │
│  │             └─────────────────────────┘                         │         │
│  └────────────────────────────┬───────────────────────────────────┘         │
│                               ▼                                              │
│                      spatial_graph.json                                      │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 2: GRAPH ANALYSIS (Optional)                                          │
│                               │                                              │
│                               ▼                                              │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                     spatial_graph.py                            │         │
│  │  • NetworkX conversion                                          │         │
│  │  • Visualization (hierarchical, depth_layers, circular)         │         │
│  │  • Queries (closer_than, farther_than)                          │         │
│  │  • Export (GraphML, GEXF)                                       │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 3: AGENTIC EVALUATION                                                 │
│                               │                                              │
│                               ▼                                              │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                      spatial_agent/                             │         │
│  │                                                                 │         │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │         │
│  │  │ annotator   │──▶│   state     │──▶│   tools     │           │         │
│  │  │             │   │             │   │             │           │         │
│  │  │ SceneConfig │   │ AgentState  │   │ LangChain   │           │         │
│  │  │ + Image     │   │ + History   │   │ Tools       │           │         │
│  │  └─────────────┘   └─────────────┘   └──────┬──────┘           │         │
│  │                                             │                   │         │
│  │                                             ▼                   │         │
│  │                                    ┌─────────────┐              │         │
│  │                                    │   agent     │              │         │
│  │                                    │             │              │         │
│  │                                    │  LangGraph  │              │         │
│  │                                    │  ReAct Loop │              │         │
│  │                                    └─────────────┘              │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                               │                                              │
│                               ▼                                              │
│                      Evaluation Results                                      │
│                   (path, success, reasoning)                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Perception Pipeline

### 1.1 Input Requirements

```
Inputs:
├── Image (JPG/PNG)
│   └── Any scene with identifiable objects
│
└── Text Prompts (List[str])
    └── Object categories to detect: ["table", "chair", "person", ...]
```

### 1.2 SAM3 Segmentation (`run_sam3.py`)

**Purpose:** Text-prompted instance segmentation using Segment Anything Model 3.

```
┌─────────────────────────────────────────────────────────────────┐
│                        SAM3 Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Image + Prompt ("table")                                       │
│           │                                                      │
│           ▼                                                      │
│   ┌───────────────────┐                                          │
│   │   Sam3Processor   │  (Text tokenization + Image preprocessing)│
│   └─────────┬─────────┘                                          │
│             │                                                    │
│             ▼                                                    │
│   ┌───────────────────┐                                          │
│   │   SAM3 Model      │  (Vision-Language Segmentation)          │
│   │   (HuggingFace)   │                                          │
│   └─────────┬─────────┘                                          │
│             │                                                    │
│             ▼                                                    │
│   ┌───────────────────┐                                          │
│   │  Post-processing  │                                          │
│   │  • NMS filtering  │                                          │
│   │  • Score threshold│                                          │
│   │  • Mask cleanup   │                                          │
│   └─────────┬─────────┘                                          │
│             │                                                    │
│             ▼                                                    │
│   Output per prompt:                                             │
│   ├── masks: np.ndarray (N, 1, H, W)  # Binary masks             │
│   ├── boxes: np.ndarray (N, 4)        # [x1, y1, x2, y2]         │
│   └── scores: np.ndarray (N,)         # Confidence scores        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Code Flow:**
```python
# run_sam3.py
class SAM3Runner:
    def __init__(self, confidence_threshold=0.3, device="cuda"):
        self.model = build_sam3_image_model("facebook/sam3-hiera-large")
        self.processor = Sam3Processor(bpe_path)

    def segment(self, image: Image, prompt: str) -> SegmentationResult:
        # 1. Process inputs
        inputs = self.processor(image, prompt)

        # 2. Run model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3. Post-process
        masks, boxes, scores = self.processor.post_process(outputs)

        # 4. Filter by confidence
        valid = scores >= self.confidence_threshold

        return SegmentationResult(
            prompt=prompt,
            masks=masks[valid],
            boxes=boxes[valid],
            scores=scores[valid]
        )
```

### 1.3 Depth Extraction (`extract_depth.py`)

**Purpose:** Monocular depth estimation from a single image.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Depth Extraction Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Backend Options:                                               │
│   ┌─────────────────────┐    ┌─────────────────────┐            │
│   │        DPT          │    │       DINOv3        │            │
│   │  (Intel/dpt-large)  │    │  (Local weights)    │            │
│   │                     │    │                     │            │
│   │  • HuggingFace      │    │  • Requires setup   │            │
│   │  • Easy setup       │    │  • Higher quality   │            │
│   │  • Recommended      │    │  • Slower           │            │
│   └──────────┬──────────┘    └──────────┬──────────┘            │
│              │                          │                        │
│              └────────────┬─────────────┘                        │
│                           ▼                                      │
│              ┌─────────────────────────┐                         │
│              │      Image Input        │                         │
│              │      (H × W × 3)        │                         │
│              └────────────┬────────────┘                         │
│                           │                                      │
│                           ▼                                      │
│              ┌─────────────────────────┐                         │
│              │   Depth Prediction      │                         │
│              │   (H × W) float32       │                         │
│              └────────────┬────────────┘                         │
│                           │                                      │
│                           ▼                                      │
│              Output:                                             │
│              ├── depth_map: np.ndarray (H, W)                    │
│              ├── depth_tensor: torch.Tensor                      │
│              └── metadata: {min, max, mean, std}                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Code Flow:**
```python
# extract_depth.py
class DPTDepthExtractor:
    def __init__(self, model_name="Intel/dpt-large"):
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.processor = DPTImageProcessor.from_pretrained(model_name)

    def extract(self, image: Image) -> DepthResult:
        # 1. Preprocess
        inputs = self.processor(image, return_tensors="pt")

        # 2. Predict depth
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # 3. Resize to original
        depth_map = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic"
        )

        return DepthResult(
            depth_map=depth_map.numpy(),
            metadata={"min": ..., "max": ..., "mean": ...}
        )
```

### 1.4 Combined Pipeline (`depth_sam3_connector.py`)

**Purpose:** Fuse segmentation and depth to produce z-ordered spatial graph.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Depth-SAM3 Connector Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Inputs:                                                        │
│   ├── image_path: str                                            │
│   └── prompts: ["table", "chair", "person", ...]                 │
│                                                                  │
│   Step 1: Parallel Extraction                                    │
│   ┌─────────────────────┐    ┌─────────────────────┐            │
│   │   SAM3 Segmentation │    │   Depth Extraction  │            │
│   │                     │    │                     │            │
│   │   For each prompt:  │    │   Full image:       │            │
│   │   → masks, boxes    │    │   → depth_map       │            │
│   └──────────┬──────────┘    └──────────┬──────────┘            │
│              │                          │                        │
│              └────────────┬─────────────┘                        │
│                           ▼                                      │
│   Step 2: Depth Statistics per Entity                            │
│   ┌─────────────────────────────────────────────────┐            │
│   │  For each detected entity:                      │            │
│   │                                                 │            │
│   │    mask = entity.mask  # (H, W) binary          │            │
│   │    masked_depth = depth_map[mask]               │            │
│   │                                                 │            │
│   │    stats = {                                    │            │
│   │        "mean": masked_depth.mean(),             │            │
│   │        "median": np.median(masked_depth),  ◄────── Key metric│
│   │        "min": masked_depth.min(),               │            │
│   │        "max": masked_depth.max(),               │            │
│   │        "std": masked_depth.std(),               │            │
│   │        "pixel_count": mask.sum()                │            │
│   │    }                                            │            │
│   └─────────────────────────────────────────────────┘            │
│                           │                                      │
│                           ▼                                      │
│   Step 3: Z-Order Ranking                                        │
│   ┌─────────────────────────────────────────────────┐            │
│   │  # Sort by median depth (lower = closer)        │            │
│   │  entities.sort(key=lambda e: e.depth_median)    │            │
│   │                                                 │            │
│   │  # Assign z-order                               │            │
│   │  for i, entity in enumerate(entities):          │            │
│   │      entity.z_order = i                         │            │
│   │                                                 │            │
│   │  # Compute relative depth [0.0, 1.0]            │            │
│   │  min_d = entities[0].depth_median               │            │
│   │  max_d = entities[-1].depth_median              │            │
│   │  for entity in entities:                        │            │
│   │      entity.relative_depth = normalize(...)     │            │
│   └─────────────────────────────────────────────────┘            │
│                           │                                      │
│                           ▼                                      │
│   Output: spatial_graph.json                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.5 Spatial Graph JSON Schema

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
      "z_order": 0,                    // 0 = closest to camera
      "relative_depth": 0.0,           // 0.0 = closest, 1.0 = farthest
      "depth_stats": {
        "mean": 7.68,
        "min": 5.43,
        "max": 9.87,
        "std": 0.73,
        "median": 7.52,                // Used for z-order ranking
        "pixel_count": 259026
      }
    },
    // ... more entities sorted by z_order
  ],

  "z_order_sequence": ["entity_0", "entity_1", ...],  // Front to back

  "metadata": {
    "num_entities": 9,
    "depth_backend": "dpt",
    "prompts": ["table", "chair", "person"],
    "timestamp": "2024-..."
  }
}
```

---

## Stage 2: Graph Analysis (Optional)

### 2.1 NetworkX Conversion (`spatial_graph.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    SpatialSceneGraph Class                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   spatial_graph.json                                             │
│           │                                                      │
│           ▼                                                      │
│   SpatialSceneGraph.from_json(path)                              │
│           │                                                      │
│           ▼                                                      │
│   ┌───────────────────────────────────────────────┐              │
│   │  NetworkX DiGraph                             │              │
│   │                                               │              │
│   │  Nodes:                                       │              │
│   │  ├── entity_0 {name, z_order, depth, ...}     │              │
│   │  ├── entity_1 {name, z_order, depth, ...}     │              │
│   │  └── ...                                      │              │
│   │                                               │              │
│   │  (No edges - z-order only, no relations)      │              │
│   └───────────────────────────────────────────────┘              │
│                                                                  │
│   Available Methods:                                             │
│   ├── get_z_ordered_entities() → List[Dict]                      │
│   ├── get_entities_closer_than(entity_id) → List[Dict]           │
│   ├── get_entities_farther_than(entity_id) → List[Dict]          │
│   ├── visualize(output_path, viz_type)                           │
│   └── export(format, path)  # graphml, gexf, json                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Visualization Types

```
┌─────────────────────────────────────────────────────────────────┐
│                     Visualization Options                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  hierarchical (default)         depth_layers                     │
│  ┌─────────────────────┐       ┌─────────────────────┐          │
│  │    ○ person (z=0)   │       │  ○         ○        │ z=0      │
│  │         │           │       │       ○             │ z=1      │
│  │    ○ lamp_0 (z=1)   │       │    ○       ○        │ z=2      │
│  │         │           │       │  ○      ○      ○    │ z=3      │
│  │    ○ table (z=2)    │       │       ○             │ z=4      │
│  │         │           │       └─────────────────────┘          │
│  │    ○ stool (z=3)    │       (X = image position,             │
│  └─────────────────────┘        Y = depth)                      │
│  (Top = close, Bottom = far)                                     │
│                                                                  │
│  circular                       spring                           │
│  ┌─────────────────────┐       ┌─────────────────────┐          │
│  │      ○     ○        │       │    ○   ○            │          │
│  │    ○         ○      │       │  ○       ○   ○      │          │
│  │                     │       │      ○       ○      │          │
│  │    ○         ○      │       │    ○     ○          │          │
│  │      ○     ○        │       └─────────────────────┘          │
│  └─────────────────────┘       (Force-directed layout)          │
│  (Arranged in circle)                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 3: Agentic Spatial Reasoning

### 3.1 System Overview

The spatial agent evaluates whether AI models can reason about 3D spatial relationships by navigating through a scene using z-ordered waypoints.

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION OBJECTIVE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Given:                                                         │
│   • An image with objects at different depths                    │
│   • Agent starting position (z=0, closest)                       │
│   • Target position (z=N, farthest)                              │
│   • Waypoints at intermediate depths                             │
│                                                                  │
│   Test:                                                          │
│   • Can the agent understand z-order from visual input?          │
│   • Can it reason about occlusion (objects blocking paths)?      │
│   • Can it plan a valid path respecting depth ordering?          │
│                                                                  │
│   Key Design:                                                    │
│   • System does NOT enforce valid moves                          │
│   • Agent must REASON about what moves are valid                 │
│   • This tests true spatial understanding                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component: `annotator.py`

**Purpose:** Generate annotated images with visual markers for waypoints, agent, and target.

```
┌─────────────────────────────────────────────────────────────────┐
│                      SpatialAnnotator                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: spatial_graph.json                                      │
│                                                                  │
│   Classes:                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  @dataclass Waypoint                                    │   │
│   │  ├── id: str              # "entity_0"                  │   │
│   │  ├── name: str            # "table_0"                   │   │
│   │  ├── category: str        # "table"                     │   │
│   │  ├── position: (x, y)     # Center in image             │   │
│   │  ├── z_order: int         # 0 = closest                 │   │
│   │  ├── relative_depth: float # 0.0 to 1.0                 │   │
│   │  └── bbox: [x1,y1,x2,y2]                                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  @dataclass SceneConfig                                 │   │
│   │  ├── image_path: str                                    │   │
│   │  ├── waypoints: List[Waypoint]   # Intermediate points  │   │
│   │  ├── agent_position: Waypoint    # Start (z=0)          │   │
│   │  ├── target_position: Waypoint   # Goal (z=N)           │   │
│   │  └── image_size: (H, W)                                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Visual Markers:                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                         │   │
│   │    ▲ AGENT (z=0)    Green triangle, starting position   │   │
│   │                                                         │   │
│   │    ● Waypoint       Circle, color = depth gradient      │   │
│   │      z=1            (Green = near, Red = far)           │   │
│   │                                                         │   │
│   │    ● Waypoint       Each labeled with "z=N"             │   │
│   │      z=2                                                │   │
│   │                                                         │   │
│   │    ★ TARGET (z=8)   Gold star, goal position            │   │
│   │                                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Methods:                                                       │
│   ├── create_scene_config(agent_z, target_z) → SceneConfig       │
│   ├── annotate(scene_config, output_path) → PIL.Image            │
│   └── generate_scenario(agent_z, target_z) → (Image, Config)     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Code Flow:**
```python
# annotator.py
annotator = SpatialAnnotator("spatial_graph.json")

# Creates SceneConfig separating agent, target, and other waypoints
config = annotator.create_scene_config(agent_z=0, target_z=8)

# Draws on image:
# 1. Waypoints (circles) - back to front for proper layering
# 2. Target (star)
# 3. Agent (triangle) - drawn last, on top
annotator.annotate(config, "annotated_scene.png")
```

### 3.3 Component: `state.py`

**Purpose:** Track agent position, path history, and view transformations.

```
┌─────────────────────────────────────────────────────────────────┐
│                         AgentState                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  @dataclass WaypointView                                │   │
│   │  ├── rotation: float       # 0-360 degrees              │   │
│   │  └── scale: float          # 0.25x to 4.0x              │   │
│   │                                                         │   │
│   │  Methods:                                               │   │
│   │  ├── rotate(angle) → WaypointView                       │   │
│   │  ├── set_scale(factor) → WaypointView                   │   │
│   │  ├── get_view_description() → str                       │   │
│   │  └── transform_position(pos, depth) → (x, y, depth')    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  @dataclass AgentState                                  │   │
│   │                                                         │   │
│   │  Position:                                              │   │
│   │  ├── current_waypoint_id: str      # "entity_8"         │   │
│   │  ├── current_z_order: int          # 0                  │   │
│   │  └── current_position: (x, y)                           │   │
│   │                                                         │   │
│   │  Goal:                                                  │   │
│   │  ├── target_waypoint_id: str       # "entity_4"         │   │
│   │  └── target_z_order: int           # 8                  │   │
│   │                                                         │   │
│   │  World:                                                 │   │
│   │  └── waypoints: Dict[id → info]    # All waypoints      │   │
│   │                                                         │   │
│   │  History:                                               │   │
│   │  ├── path_history: List[str]       # IDs visited        │   │
│   │  └── move_count: int                                    │   │
│   │                                                         │   │
│   │  View:                                                  │   │
│   │  └── view: WaypointView                                 │   │
│   │                                                         │   │
│   │  Status:                                                │   │
│   │  └── reached_target: bool                               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Key Methods:                                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  move_to(waypoint_id) → (success: bool, message: str)   │   │
│   │                                                         │   │
│   │  IMPORTANT: Does NOT validate occlusion!                │   │
│   │  The agent must reason about valid moves itself.        │   │
│   │  This is the core evaluation mechanism.                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Other Methods:                                                 │
│   ├── rotate_view(angle) → str                                  │
│   ├── scale_view(factor) → str                                  │
│   ├── get_all_waypoints() → List[Dict]  (sorted by z)           │
│   ├── get_waypoints_between(z1, z2) → List[Dict]                │
│   ├── get_status() → str                                        │
│   └── get_path_summary() → Dict                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**State Transitions:**
```
Initial State:
┌────────────────────────────┐
│ current: entity_8 (z=0)    │
│ target: entity_4 (z=8)     │
│ path_history: ["entity_8"] │
│ move_count: 0              │
│ reached_target: False      │
└────────────────────────────┘
            │
            │ move_to("entity_5")
            ▼
┌────────────────────────────┐
│ current: entity_5 (z=1)    │
│ target: entity_4 (z=8)     │
│ path_history: ["entity_8", │
│               "entity_5"]  │
│ move_count: 1              │
│ reached_target: False      │
└────────────────────────────┘
            │
            │ ... more moves ...
            ▼
┌────────────────────────────┐
│ current: entity_4 (z=8)    │
│ target: entity_4 (z=8)     │
│ path_history: [...]        │
│ move_count: 4              │
│ reached_target: True ✓     │
└────────────────────────────┘
```

### 3.4 Component: `tools.py`

**Purpose:** LangChain tool interfaces for agent actions.

```
┌─────────────────────────────────────────────────────────────────┐
│                       LangChain Tools                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  get_waypoints                                            │  │
│  │  ─────────────                                            │  │
│  │  Input:  include_current: bool = True                     │  │
│  │  Output: List of waypoints with z-order, position         │  │
│  │                                                           │  │
│  │  Purpose: Understand scene layout before planning         │  │
│  │                                                           │  │
│  │  Example Output:                                          │  │
│  │  "Available waypoints (sorted by z-order):                │  │
│  │   - entity_8: person [CURRENT]                            │  │
│  │       z_order: 0, depth: 0.000                            │  │
│  │       position: (4350, 2473)                              │  │
│  │   - entity_5: lamp_0                                      │  │
│  │       z_order: 1, depth: 0.233                            │  │
│  │   ..."                                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  move_to                                                  │  │
│  │  ───────                                                  │  │
│  │  Input:  waypoint_id: str  (e.g., "entity_3")             │  │
│  │  Output: SUCCESS/FAILED + new status                      │  │
│  │                                                           │  │
│  │  Purpose: Navigate to a waypoint                          │  │
│  │                                                           │  │
│  │  Note: Always succeeds if waypoint exists.                │  │
│  │        Agent must reason about occlusion BEFORE calling.  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  rotate                                                   │  │
│  │  ──────                                                   │  │
│  │  Input:  angle: float  (degrees, + = right, - = left)     │  │
│  │  Output: View description + transformed positions         │  │
│  │                                                           │  │
│  │  Purpose: See depth relationships from different angle    │  │
│  │                                                           │  │
│  │  Example: rotate(90) shows side view, revealing which     │  │
│  │           objects are truly behind others                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  scale                                                    │  │
│  │  ─────                                                    │  │
│  │  Input:  factor: float  (>1 = zoom in, <1 = zoom out)     │  │
│  │  Output: View description + depth gaps                    │  │
│  │                                                           │  │
│  │  Purpose: See fine depth differences between waypoints    │  │
│  │                                                           │  │
│  │  Example: scale(2.0) zooms in, showing gaps between       │  │
│  │           consecutive z-orders more clearly               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tool Implementation Pattern:**
```python
class MoveToTool(BaseTool):
    name: str = "move_to"
    description: str = """Move to a specific waypoint.

    IMPORTANT: Before moving, reason about whether the path is valid:
    - Consider z-order: moving through objects may be blocked
    - Consider occlusion: objects in front block access
    """
    args_schema: Type[BaseModel] = MoveToInput
    agent_state: AgentState = None

    def _run(self, waypoint_id: str, ...) -> str:
        success, message = self.agent_state.move_to(waypoint_id)
        return f"{'SUCCESS' if success else 'FAILED'}: {message}\n\n{self.agent_state.get_status()}"
```

### 3.5 Component: `agent.py`

**Purpose:** LangGraph ReAct agent orchestration.

```
┌─────────────────────────────────────────────────────────────────┐
│                   SpatialReasoningAgent                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Initialization:                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  agent = SpatialReasoningAgent(                         │   │
│   │      model_name="claude-sonnet-4-20250514",             │   │
│   │      temperature=0.0,                                   │   │
│   │      verbose=True                                       │   │
│   │  )                                                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Setup:                                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  config = agent.setup_scenario(                         │   │
│   │      spatial_graph_path="spatial_graph.json",           │   │
│   │      agent_z=0,                                         │   │
│   │      target_z=8,                                        │   │
│   │      output_dir="outputs/"                              │   │
│   │  )                                                      │   │
│   │                                                         │   │
│   │  Internal Steps:                                        │   │
│   │  1. SpatialAnnotator(graph_path)                        │   │
│   │  2. create_scene_config(agent_z, target_z)              │   │
│   │  3. annotate() → saves annotated image                  │   │
│   │  4. AgentState.from_scene_config(config)                │   │
│   │  5. create_tools(agent_state)                           │   │
│   │  6. create_react_agent(llm, tools, prompt)              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Execution Modes:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                         │   │
│   │  Mode 1: run() - LangGraph ReAct                        │   │
│   │  ────────────────────────────────                       │   │
│   │  Uses create_react_agent with tool calling              │   │
│   │  Text-only input (no image)                             │   │
│   │                                                         │   │
│   │  Mode 2: run_with_image() - Multimodal                  │   │
│   │  ──────────────────────────────────────                 │   │
│   │  Direct Anthropic API with image + text                 │   │
│   │  Manual tool parsing loop                               │   │
│   │  Agent sees annotated scene image                       │   │
│   │                                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.6 System Prompt

```
┌─────────────────────────────────────────────────────────────────┐
│                        SYSTEM PROMPT                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  You are an embodied spatial reasoning agent navigating         │
│  through a 3D scene.                                             │
│                                                                  │
│  ## Your Situation                                               │
│  - You are positioned at a specific location (marked as AGENT)  │
│  - You need to navigate to a target (marked as TARGET/star)     │
│  - The scene contains waypoints at different depths (z-orders)  │
│  - z=0 is closest to camera, higher z values are farther        │
│                                                                  │
│  ## Your Tools                                                   │
│  1. get_waypoints: See all waypoints with z-order and positions │
│  2. move_to: Move to a specific waypoint                         │
│  3. rotate: Rotate view to understand spatial relationships     │
│  4. scale: Zoom to see depth relationships more clearly         │
│                                                                  │
│  ## Key Concepts                                                 │
│  - Z-order: Lower z = closer, higher z = farther                │
│  - Occlusion: Objects in front can block access to behind       │
│  - Valid paths: Must not pass through obstacles                 │
│                                                                  │
│  ## Your Task                                                    │
│  1. Use get_waypoints to understand the scene layout            │
│  2. Analyze z-order relationships and identify paths            │
│  3. Consider using rotate/scale for better understanding        │
│  4. Plan a path from current position to target                 │
│  5. Execute the path using move_to commands                     │
│  6. Explain your reasoning at each step                         │
│                                                                  │
│  ## Important                                                    │
│  - Think carefully about occlusion                              │
│  - Consider z-order when planning your path                     │
│  - Explain why you chose your path                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.7 Execution Flow (Multimodal Mode)

```
┌─────────────────────────────────────────────────────────────────┐
│                    run_with_image() Flow                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. INITIAL MESSAGE                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  [Image: annotated_scene.png (base64)]                  │   │
│   │                                                         │   │
│   │  "Look at this annotated scene image.                   │   │
│   │   You are the AGENT (green triangle).                   │   │
│   │   Navigate to TARGET (yellow star).                     │   │
│   │                                                         │   │
│   │   Current position: person (z=0)                        │   │
│   │   Target: stool_2 (z=8)                                 │   │
│   │                                                         │   │
│   │   To use a tool, respond with:                          │   │
│   │   TOOL: <name>                                          │   │
│   │   ARGS: <json>"                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│   2. CONVERSATION LOOP (max 15 turns)                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                         │   │
│   │   ┌─────────┐                                           │   │
│   │   │  Claude │ ──► Response with reasoning               │   │
│   │   └────┬────┘                                           │   │
│   │        │                                                │   │
│   │        ▼                                                │   │
│   │   Contains "TOOL:"?                                     │   │
│   │        │                                                │   │
│   │   ┌────┴────┐                                           │   │
│   │   │ Yes     │ No                                        │   │
│   │   ▼         ▼                                           │   │
│   │  Parse    Prompt for                                    │   │
│   │  tool     action                                        │   │
│   │   │                                                     │   │
│   │   ▼                                                     │   │
│   │  Execute tool                                           │   │
│   │  (move_to, get_waypoints, etc.)                         │   │
│   │   │                                                     │   │
│   │   ▼                                                     │   │
│   │  Return result                                          │   │
│   │   │                                                     │   │
│   │   ▼                                                     │   │
│   │  reached_target? ──► Yes ──► Exit loop                  │   │
│   │        │                                                │   │
│   │       No                                                │   │
│   │        │                                                │   │
│   │        └──────────► Continue loop                       │   │
│   │                                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│   3. OUTPUT                                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  {                                                      │   │
│   │    "conversation": [...],                               │   │
│   │    "path_summary": {                                    │   │
│   │      "path": [                                          │   │
│   │        {"id": "entity_8", "name": "person", "z": 0},    │   │
│   │        {"id": "entity_5", "name": "lamp_0", "z": 1},    │   │
│   │        ...                                              │   │
│   │      ],                                                 │   │
│   │      "total_moves": 4                                   │   │
│   │    },                                                   │   │
│   │    "reached_target": true,                              │   │
│   │    "total_moves": 4,                                    │   │
│   │    "final_state": {...}                                 │   │
│   │  }                                                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete End-to-End Example

```bash
# Step 1: Run perception pipeline
python depth_sam3_connector.py \
    --image scene.jpg \
    --prompts "table" "chair" "person" "lamp" \
    --output_dir spatial_outputs/

# Output: spatial_outputs/scene_spatial_graph.json

# Step 2: (Optional) Visualize graph
python spatial_graph.py \
    --input spatial_outputs/scene_spatial_graph.json \
    --viz --viz_type hierarchical

# Step 3: Run spatial agent evaluation
ANTHROPIC_API_KEY=xxx python run_spatial_agent.py \
    --graph spatial_outputs/scene_spatial_graph.json \
    --agent-z 0 \
    --target-z 8 \
    --multimodal \
    --output-dir spatial_agent_outputs/

# Output:
# - spatial_agent_outputs/scenario_annotated.png
# - Console: Agent reasoning and path taken
```

**Expected Output:**
```
=== Scenario Setup ===
Agent: person (z=0)
Target: stool_2 (z=8)
Waypoints: 7

=== Turn 1 ===
I'll first examine the available waypoints to understand the scene layout.

TOOL: get_waypoints
ARGS: {"include_current": true}

[Tool Result]
Available waypoints (sorted by z-order):
- entity_8: person [CURRENT] z=0
- entity_5: lamp_0 z=1
- entity_1: table_1 z=2
...

=== Turn 2 ===
I can see the z-ordering. To reach the target at z=8, I need to navigate
through intermediate waypoints. Let me move to the next waypoint...

TOOL: move_to
ARGS: {"waypoint_id": "entity_5"}

...

=== Results ===
Reached target: True
Total moves: 4
Path: entity_8 → entity_5 → entity_1 → entity_0 → entity_4
```

---

## File Reference

```
cherry/
├── depth_sam3_connector.py   # Stage 1: Perception pipeline
├── run_sam3.py               # SAM3 segmentation
├── extract_depth.py          # Depth extraction
├── spatial_graph.py          # Stage 2: Graph analysis
├── run_spatial_agent.py      # Stage 3: Agent runner
│
├── spatial_agent/
│   ├── __init__.py           # Package exports
│   ├── annotator.py          # Image annotation
│   ├── state.py              # Agent state management
│   ├── tools.py              # LangChain tools
│   └── agent.py              # LangGraph agent
│
├── sam3/                     # Submodule: SAM3 model
├── dinov3/                   # Submodule: DINOv3 depth
│
├── requirements.txt          # Dependencies
├── README.md                 # User documentation
└── PIPELINE_FLOW.md          # This file
```
