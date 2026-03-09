#!/bin/bash
# generate_all_spatial_graphs.sh
# Generate spatial graphs for all images in visual_scenes_2d_unsplash
#
# Run on ml-login7:
#   chmod +x generate_all_spatial_graphs.sh
#   ./generate_all_spatial_graphs.sh
#
# Requirements:
#   - GPU for SAM3 + Depth
#   - OpenAI API key for object detection

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================

# Initialize conda
source /home/lauren/miniconda3/etc/profile.d/conda.sh
conda activate cherry

# Load API keys
if [ -f ~/cherry/.env ]; then
    export OPENAI_API_KEY=$(grep OPENAI_API_KEY ~/cherry/.env | cut -d'=' -f2 | tr -d ' ')
fi

# Set paths
export PYTHONPATH=/home/lauren/cherry/sam3:$PYTHONPATH
cd /home/lauren/cherry

# Output directory
OUTPUT_DIR="spatial_outputs"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="${OUTPUT_DIR}/generation_log_$(date +%Y%m%d_%H%M%S).txt"

# GPU to use (adjust as needed)
GPU_ID=0

# ============================================================
# Main Processing
# ============================================================

echo "============================================================" | tee "$LOG_FILE"
echo "Spatial Graph Generation - $(date)" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Find all images
mapfile -t IMAGES < <(find visual_scenes_2d_unsplash -name "*.jpg" -type f | sort)
TOTAL=${#IMAGES[@]}

echo "Found $TOTAL images to process" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track results
PROCESSED=0
SKIPPED=0
FAILED=0

for i in "${!IMAGES[@]}"; do
    IMAGE="${IMAGES[$i]}"
    NAME=$(basename "$IMAGE" .jpg)
    GRAPH_FILE="${OUTPUT_DIR}/${NAME}_spatial_graph.json"

    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "[$((i+1))/$TOTAL] $NAME" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

    # Skip if already processed
    if [ -f "$GRAPH_FILE" ]; then
        echo "  SKIP: Spatial graph already exists" | tee -a "$LOG_FILE"
        ((SKIPPED++))
        continue
    fi

    # Create temp directory for intermediate files
    TEMP_DIR="${OUTPUT_DIR}/temp_${NAME}"
    mkdir -p "$TEMP_DIR"

    # Step 1: Object Detection using GPT-4o
    echo "  Step 1: Detecting objects..." | tee -a "$LOG_FILE"
    PROMPTS_FILE="${TEMP_DIR}/prompts.json"

    if ! python generate_queries.py \
        --image "$IMAGE" \
        --auto-detect-objects \
        --output "${TEMP_DIR}/detection_output.json" \
        >> "$LOG_FILE" 2>&1; then
        echo "  FAILED: Object detection failed" | tee -a "$LOG_FILE"
        ((FAILED++))
        rm -rf "$TEMP_DIR"
        continue
    fi

    # Extract prompts from detection output
    # The prompts.json should be created by generate_queries.py
    if [ ! -f "$PROMPTS_FILE" ] && [ -f "${TEMP_DIR}/detection_output.json" ]; then
        # Extract object names from detection output
        python -c "
import json
with open('${TEMP_DIR}/detection_output.json') as f:
    data = json.load(f)
prompts = data.get('detected_objects', data.get('prompts', []))
if isinstance(prompts, dict):
    prompts = list(prompts.keys())
with open('${PROMPTS_FILE}', 'w') as f:
    json.dump(prompts, f)
print(f'Extracted {len(prompts)} prompts')
" >> "$LOG_FILE" 2>&1
    fi

    if [ ! -f "$PROMPTS_FILE" ]; then
        echo "  FAILED: No prompts file generated" | tee -a "$LOG_FILE"
        ((FAILED++))
        rm -rf "$TEMP_DIR"
        continue
    fi

    echo "  Step 1: Done" | tee -a "$LOG_FILE"

    # Step 2: SAM3 + Depth
    echo "  Step 2: Running SAM3 + Depth..." | tee -a "$LOG_FILE"

    if ! CUDA_VISIBLE_DEVICES=$GPU_ID python depth_sam3_connector.py \
        --image "$IMAGE" \
        --prompts-file "$PROMPTS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --no_masks \
        >> "$LOG_FILE" 2>&1; then
        echo "  FAILED: SAM3 + Depth failed" | tee -a "$LOG_FILE"
        ((FAILED++))
        rm -rf "$TEMP_DIR"
        continue
    fi

    # Verify spatial graph was created
    if [ ! -f "$GRAPH_FILE" ]; then
        echo "  FAILED: Spatial graph not created" | tee -a "$LOG_FILE"
        ((FAILED++))
        rm -rf "$TEMP_DIR"
        continue
    fi

    # Count objects in graph
    NUM_OBJECTS=$(python -c "import json; print(len(json.load(open('$GRAPH_FILE'))['nodes']))" 2>/dev/null || echo "?")

    echo "  Step 2: Done ($NUM_OBJECTS objects detected)" | tee -a "$LOG_FILE"
    echo "  SUCCESS: $GRAPH_FILE" | tee -a "$LOG_FILE"

    # Cleanup temp directory
    rm -rf "$TEMP_DIR"

    ((PROCESSED++))
    echo "" | tee -a "$LOG_FILE"
done

# ============================================================
# Summary
# ============================================================

echo "============================================================" | tee -a "$LOG_FILE"
echo "GENERATION COMPLETE - $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Total images: $TOTAL" | tee -a "$LOG_FILE"
echo "Processed:    $PROCESSED" | tee -a "$LOG_FILE"
echo "Skipped:      $SKIPPED (already existed)" | tee -a "$LOG_FILE"
echo "Failed:       $FAILED" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# List all spatial graphs
echo "Spatial graphs in $OUTPUT_DIR:" | tee -a "$LOG_FILE"
ls -la "$OUTPUT_DIR"/*_spatial_graph.json 2>/dev/null | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
