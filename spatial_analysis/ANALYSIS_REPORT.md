# Qwen2.5-VL-7B Spatial Reasoning Analysis Report

## Executive Summary

We conducted mechanistic interpretability analysis on Qwen2.5-VL-7B-Instruct to understand its spatial reasoning capabilities compared to closed-source VLMs (GPT-5-mini, Claude Haiku 4.5).

### Key Findings (50 Images, 3672 Queries)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **59.8%** (2195/3672) |
| Egocentric | 63.9% (1477/2312) |
| Allocentric | 52.8% (718/1360) |
| Allocentric Gap | -11.1 pp |

**Allocentric object-to-object front/behind is the hardest subcategory at 46.2% accuracy.**

---

## Model Architecture

### Qwen2.5-VL-7B-Instruct
- **Total Parameters**: 8.29B
- **Memory (bf16)**: ~16.6 GB

#### Vision Encoder
- 32 Qwen2_5_VLVisionBlock layers (676M params)
- Hidden size: 1280
- 16 attention heads
- Patch size: 14×14
- Full attention at layers: [7, 15, 23, 31]

#### Language Model
- 28 Qwen2_5_VLDecoderLayer layers (7B params)
- Hidden size: 3584
- 28 attention heads, 4 KV heads (grouped-query attention)
- RoPE with theta=1M

#### Cross-Modal
- Vision-Language Merger (45M params)
- Projects 1280 → 3584 dimensions

---

## Evaluation Results (50 Images, 3672 Queries)

### By Relation Axis

| Relation Axis | Correct | Total | Accuracy |
|---------------|---------|-------|----------|
| above_below | 500 | 764 | **65.4%** |
| left_right | 881 | 1440 | 61.2% |
| foreground_background | 239 | 396 | 60.4% |
| front_behind | 575 | 1072 | 53.6% |

### Failure Analysis

**Total Failures**: 1477/3672 (40.2%)

| Category | Failures | Share | Error Rate |
|----------|----------|-------|------------|
| left_right | 559 | 37.8% | 38.8% |
| front_behind | 497 | 33.6% | 46.4% |
| above_below | 264 | 17.9% | 34.6% |
| foreground_background | 157 | 10.6% | 39.6% |

**Breakdown by Task Type**:
- Egocentric: 835 failures (36.1% error rate)
- Allocentric: 642 failures (47.2% error rate)

### Pattern Analysis

1. **Allocentric front_behind is hardest**
   - Object-to-object: 46.2% accuracy
   - Viewer-centered: 52.6% accuracy
   - Requires mental rotation and reference frame transformation

2. **Vertical relations (above/below) are easiest**
   - Viewer-centered: 65.3% accuracy
   - Object-to-object: 65.6% accuracy
   - Consistent across frame types

3. **Frame type has minimal impact**
   - Viewer-centered: 60.1% accuracy
   - Object-to-object: 59.5% accuracy
   - The difficulty comes from task type and relation axis, not frame type

---

## Recommended Hook Points for Interpretability

### Vision Encoder
```
model.visual.blocks.0     # Early features
model.visual.blocks.7     # Full attention (low-level spatial)
model.visual.blocks.15    # Full attention (mid-level)
model.visual.blocks.23    # Full attention (high-level)
model.visual.blocks.31    # Final vision representation
```

### Cross-Modal
```
model.visual.merger       # Vision-language projection
model.visual.merger.mlp   # Projection MLP
```

### Language Model
```
model.language_model.layers.0    # Query understanding
model.language_model.layers.7    # Early reasoning
model.language_model.layers.14   # Mid-level reasoning
model.language_model.layers.21   # Late reasoning
model.language_model.layers.27   # Decision layer
```

---

## Hypotheses for Future Investigation

1. **Spatial encoding in vision blocks**
   - Probe activations at full-attention layers (7, 15, 23, 31)
   - Test if left/right is encoded earlier than front/behind

2. **Reference frame transformation**
   - Allocentric requires transforming viewer perspective
   - May occur in mid-to-late LM layers (10-21)

3. **Depth perception**
   - front_behind and foreground_background both require depth
   - Patch embeddings may not preserve depth cues effectively

4. **Cross-attention patterns**
   - Compare attention to relevant objects for success vs failure
   - Check if model attends to correct spatial regions

---

## Files Generated

```
spatial_analysis/
├── inspect_qwen_architecture.py   # Architecture inspection
├── activation_extractor.py        # Hook-based extraction
├── attention_visualizer.py        # Visualization tools
├── run_qwen_eval.py               # Baseline evaluation
├── run_analysis.py                # Full analysis pipeline
└── ANALYSIS_REPORT.md             # This report

eval_outputs/
├── 000000000139__qwen__qwen/
│   └── evaluation_results.json    # Full results

eval_visualizations/
├── results_analysis_3models.tex   # LaTeX tables
└── [visualization files]
```

---

## Conclusions

1. **Qwen2.5-VL-7B is competitive** with closed-source models on egocentric spatial reasoning

2. **Allocentric reasoning is challenging** for all models, but particularly for Qwen (-26pp vs egocentric)

3. **Front/behind relations** are the main failure mode, especially when requiring reference frame transformation

4. **Architecture enables probing** - clear hook points for mechanistic analysis at vision, merger, and LM layers

---

## Next Steps

1. Run linear probing on layer activations to identify where spatial relations are encoded
2. Extract attention patterns for success vs failure cases
3. Perform activation patching to identify causal mechanisms
4. Compare representations across relation types using RSA
