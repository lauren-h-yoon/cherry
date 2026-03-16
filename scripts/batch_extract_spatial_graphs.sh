#!/bin/bash
#
# batch_extract_spatial_graphs.sh
#
# Process all COCO val2017 images through depth_sam3_connector
# to extract spatial graphs for later query generation and evaluation.
#
# Usage:
#   ./scripts/batch_extract_spatial_graphs.sh [OPTIONS]
#
# Options:
#   --start N       Start from image index N (default: 0)
#   --limit N       Process only N images (default: all)
#   --skip-existing Skip images that already have spatial_graph.json
#

# Don't exit on error - we want to continue processing
set +e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Directories
VAL2017_DIR="val2017"
ANNOTATIONS_FILE="annotations/instances_val2017.json"
OUTPUT_DIR="spatial_outputs_coco"
LOG_DIR="logs"

# Processing options
MAX_PER_CATEGORY=3
MIN_CONFIDENCE=0.5
DEVICE="cuda"

# Parse arguments
START_INDEX=0
LIMIT=0
SKIP_EXISTING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --start)
            START_INDEX="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Get list of images
IMAGES=($(ls "$VAL2017_DIR"/*.jpg 2>/dev/null | sort))
TOTAL_IMAGES=${#IMAGES[@]}

if [ $TOTAL_IMAGES -eq 0 ]; then
    echo "ERROR: No images found in $VAL2017_DIR"
    exit 1
fi

echo "============================================================"
echo "COCO val2017 Spatial Graph Extraction"
echo "============================================================"
echo "Total images available: $TOTAL_IMAGES"
echo "Starting from index: $START_INDEX"
echo "Limit: $([ $LIMIT -eq 0 ] && echo 'all' || echo $LIMIT)"
echo "Skip existing: $SKIP_EXISTING"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Calculate end index
if [ $LIMIT -eq 0 ]; then
    END_INDEX=$TOTAL_IMAGES
else
    END_INDEX=$((START_INDEX + LIMIT))
    if [ $END_INDEX -gt $TOTAL_IMAGES ]; then
        END_INDEX=$TOTAL_IMAGES
    fi
fi

# Counters
PROCESSED=0
SKIPPED=0
FAILED=0
START_TIME=$(date +%s)

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/batch_extract_${TIMESTAMP}.log"

echo "Log file: $LOG_FILE"
echo ""

# Process images
for ((i=START_INDEX; i<END_INDEX; i++)); do
    IMAGE_PATH="${IMAGES[$i]}"
    IMAGE_NAME=$(basename "$IMAGE_PATH" .jpg)
    SPATIAL_GRAPH="$OUTPUT_DIR/${IMAGE_NAME}_spatial_graph.json"

    # Progress
    CURRENT=$((i - START_INDEX + 1))
    TOTAL=$((END_INDEX - START_INDEX))
    PERCENT=$((CURRENT * 100 / TOTAL))

    echo -n "[$CURRENT/$TOTAL] ($PERCENT%) $IMAGE_NAME ... "

    # Skip if exists
    if [ "$SKIP_EXISTING" = true ] && [ -f "$SPATIAL_GRAPH" ]; then
        echo "SKIPPED (exists)"
        ((SKIPPED++))
        continue
    fi

    # Run depth_sam3_connector
    if python depth_sam3_connector.py \
        --image "$IMAGE_PATH" \
        --prompt-source coco_gt \
        --coco-annotations "$ANNOTATIONS_FILE" \
        --max_per_category "$MAX_PER_CATEGORY" \
        --min_confidence "$MIN_CONFIDENCE" \
        --output_dir "$OUTPUT_DIR" \
        --no_masks \
        --no_depth \
        --device "$DEVICE" \
        >> "$LOG_FILE" 2>&1; then
        echo "OK"
        ((PROCESSED++))
    else
        echo "FAILED"
        ((FAILED++))
        echo "  Error processing $IMAGE_NAME" >> "$LOG_FILE"
    fi

    # Progress stats every 100 images
    if [ $((CURRENT % 100)) -eq 0 ]; then
        ELAPSED=$(($(date +%s) - START_TIME))
        RATE=$(echo "scale=2; $CURRENT / $ELAPSED" | bc 2>/dev/null || echo "N/A")
        ETA=$(echo "scale=0; ($TOTAL - $CURRENT) / $RATE" | bc 2>/dev/null || echo "N/A")
        echo ""
        echo "  Progress: $PROCESSED processed, $SKIPPED skipped, $FAILED failed"
        echo "  Rate: $RATE images/sec, ETA: ${ETA}s"
        echo ""
    fi
done

# Final summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Batch Processing Complete"
echo "============================================================"
echo "Processed: $PROCESSED"
echo "Skipped: $SKIPPED"
echo "Failed: $FAILED"
echo "Total time: ${ELAPSED}s"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "============================================================"

# Count output files
NUM_GRAPHS=$(ls "$OUTPUT_DIR"/*_spatial_graph.json 2>/dev/null | wc -l)
echo "Spatial graphs generated: $NUM_GRAPHS"
