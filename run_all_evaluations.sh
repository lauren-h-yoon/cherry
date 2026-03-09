#!/bin/bash
# run_all_evaluations.sh
# Run embodied spatial evaluations across all images and models
#
# Run on ml-login7:
#   chmod +x run_all_evaluations.sh
#   ./run_all_evaluations.sh
#
# Prerequisites:
#   - Spatial graphs generated for all images (run generate_all_spatial_graphs.sh first)
#   - vLLM server running for Qwen model (optional)
#   - API keys for Claude and OpenAI

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================

# Initialize conda
source /home/lauren/miniconda3/etc/profile.d/conda.sh
conda activate cherry

# Load API keys
if [ -f ~/cherry/.env ]; then
    export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY ~/cherry/.env | cut -d'=' -f2 | tr -d ' ')
    export OPENAI_API_KEY=$(grep OPENAI_API_KEY ~/cherry/.env | cut -d'=' -f2 | tr -d ' ')
fi

# Set paths
cd /home/lauren/cherry

# Configuration
SPATIAL_OUTPUTS="spatial_outputs"
EVAL_RESULTS="eval_results"
NUM_TASKS=20  # 5 per task type × 4 types
SEED=42

# vLLM configuration (if using)
VLLM_URL="http://localhost:8000/v1"
QWEN_MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Models to evaluate
# Format: "provider:model_name:output_dir_name"
MODELS=(
    "vllm:${QWEN_MODEL}:qwen3vl_8b"
    "claude:claude-3-haiku-20240307:claude_haiku"
    "openai:gpt-5-mini:gpt5_mini"
)

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${EVAL_RESULTS}/logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

MAIN_LOG="${LOG_DIR}/evaluation_log.txt"

# ============================================================
# Helper Functions
# ============================================================

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

run_evaluation() {
    local PROVIDER=$1
    local MODEL=$2
    local OUTPUT_NAME=$3
    local GRAPH=$4
    local IMAGE=$5
    local IMAGE_NAME=$6

    local OUTPUT_DIR="${EVAL_RESULTS}/${OUTPUT_NAME}/${IMAGE_NAME}"
    mkdir -p "$OUTPUT_DIR"

    local CMD="python run_embodied_eval.py \
        --graph \"$GRAPH\" \
        --image \"$IMAGE\" \
        --provider $PROVIDER \
        --task suite \
        --num-tasks $NUM_TASKS \
        --seed $SEED \
        --output-dir \"$OUTPUT_DIR\""

    # Add model-specific options
    if [ "$PROVIDER" == "vllm" ]; then
        CMD="$CMD --vllm-url $VLLM_URL --model $MODEL"
    elif [ "$PROVIDER" != "claude" ]; then
        CMD="$CMD --model $MODEL"
    fi

    # Run evaluation
    if eval $CMD >> "${OUTPUT_DIR}/run.log" 2>&1; then
        return 0
    else
        return 1
    fi
}

# ============================================================
# Pre-flight Checks
# ============================================================

log "============================================================"
log "Embodied Spatial Evaluation - Multi-Model Run"
log "============================================================"
log "Timestamp: $TIMESTAMP"
log "Tasks per image: $NUM_TASKS"
log "Seed: $SEED"
log ""

# Check spatial graphs exist
GRAPHS=($(find "$SPATIAL_OUTPUTS" -name "*_spatial_graph.json" | sort))
NUM_GRAPHS=${#GRAPHS[@]}

if [ $NUM_GRAPHS -eq 0 ]; then
    log "ERROR: No spatial graphs found in $SPATIAL_OUTPUTS"
    log "Run generate_all_spatial_graphs.sh first"
    exit 1
fi

log "Found $NUM_GRAPHS spatial graphs"
log ""

# Check API keys
if [ -z "$ANTHROPIC_API_KEY" ]; then
    log "WARNING: ANTHROPIC_API_KEY not set - Claude evaluation will fail"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    log "WARNING: OPENAI_API_KEY not set - OpenAI evaluation will fail"
fi

# Check vLLM server (optional)
VLLM_AVAILABLE=false
if curl -s "${VLLM_URL}/models" > /dev/null 2>&1; then
    VLLM_AVAILABLE=true
    log "vLLM server: AVAILABLE at $VLLM_URL"
else
    log "vLLM server: NOT AVAILABLE at $VLLM_URL"
    log "  To start: vllm serve $QWEN_MODEL --enable-auto-tool-choice --tool-call-parser hermes"
fi

log ""

# ============================================================
# Main Evaluation Loop
# ============================================================

# Calculate totals
TOTAL_EVALS=$((NUM_GRAPHS * ${#MODELS[@]}))
CURRENT=0
PASSED=0
FAILED=0

log "Starting evaluations: $TOTAL_EVALS total (${NUM_GRAPHS} images × ${#MODELS[@]} models)"
log ""

for GRAPH in "${GRAPHS[@]}"; do
    IMAGE_NAME=$(basename "$GRAPH" _spatial_graph.json)

    # Get image path from graph
    IMAGE=$(python -c "import json; print(json.load(open('$GRAPH')).get('image_path', ''))" 2>/dev/null)

    if [ -z "$IMAGE" ] || [ ! -f "$IMAGE" ]; then
        # Try to find image
        IMAGE=$(find visual_scenes_2d_unsplash -name "${IMAGE_NAME}.jpg" | head -1)
    fi

    if [ -z "$IMAGE" ] || [ ! -f "$IMAGE" ]; then
        log "WARNING: Image not found for $IMAGE_NAME, skipping"
        continue
    fi

    log "------------------------------------------------------------"
    log "Image: $IMAGE_NAME"
    log "------------------------------------------------------------"

    for MODEL_SPEC in "${MODELS[@]}"; do
        IFS=':' read -r PROVIDER MODEL OUTPUT_NAME <<< "$MODEL_SPEC"
        ((CURRENT++))

        # Skip vLLM if not available
        if [ "$PROVIDER" == "vllm" ] && [ "$VLLM_AVAILABLE" == "false" ]; then
            log "  [$CURRENT/$TOTAL_EVALS] $OUTPUT_NAME: SKIPPED (vLLM not available)"
            continue
        fi

        log "  [$CURRENT/$TOTAL_EVALS] $OUTPUT_NAME: Running..."

        if run_evaluation "$PROVIDER" "$MODEL" "$OUTPUT_NAME" "$GRAPH" "$IMAGE" "$IMAGE_NAME"; then
            # Get score from results
            RESULT_FILE="${EVAL_RESULTS}/${OUTPUT_NAME}/${IMAGE_NAME}/embodied_eval_results.json"
            if [ -f "$RESULT_FILE" ]; then
                SCORE=$(python -c "import json; r=json.load(open('$RESULT_FILE')); print(f\"{r.get('summary',{}).get('overall_score',0):.1%}\")" 2>/dev/null || echo "?")
                log "  [$CURRENT/$TOTAL_EVALS] $OUTPUT_NAME: PASSED (score: $SCORE)"
            else
                log "  [$CURRENT/$TOTAL_EVALS] $OUTPUT_NAME: PASSED (no results file)"
            fi
            ((PASSED++))
        else
            log "  [$CURRENT/$TOTAL_EVALS] $OUTPUT_NAME: FAILED"
            ((FAILED++))
        fi
    done

    log ""
done

# ============================================================
# Aggregate Results
# ============================================================

log "============================================================"
log "EVALUATION COMPLETE - $(date)"
log "============================================================"
log "Total evaluations: $TOTAL_EVALS"
log "Passed: $PASSED"
log "Failed: $FAILED"
log ""

# Create aggregate results
log "Aggregating results..."

python - <<'PYEOF'
import json
import glob
import os
from collections import defaultdict

eval_results = os.environ.get('EVAL_RESULTS', 'eval_results')
timestamp = os.environ.get('TIMESTAMP', '')

# Collect all results
results_by_model = defaultdict(list)

for model_dir in glob.glob(f"{eval_results}/*/"):
    model_name = os.path.basename(model_dir.rstrip('/'))
    if model_name.startswith('logs'):
        continue

    for result_file in glob.glob(f"{model_dir}/*/embodied_eval_results.json"):
        try:
            with open(result_file) as f:
                result = json.load(f)
            image_name = os.path.basename(os.path.dirname(result_file))
            result['image'] = image_name
            results_by_model[model_name].append(result)
        except Exception as e:
            print(f"Error reading {result_file}: {e}")

# Print summary per model
print("\n" + "="*60)
print("RESULTS BY MODEL")
print("="*60)

aggregate = {
    'timestamp': timestamp,
    'models': {}
}

for model_name, results in sorted(results_by_model.items()):
    if not results:
        continue

    scores = [r.get('summary', {}).get('overall_score', 0) for r in results]
    task_scores = defaultdict(list)

    for r in results:
        for task in r.get('tasks', []):
            task_type = task.get('task_type', 'unknown')
            task_scores[task_type].append(task.get('score', 0))

    print(f"\n{model_name}:")
    print(f"  Images evaluated: {len(results)}")
    print(f"  Overall score:    {sum(scores)/len(scores):.1%} (avg)")
    print(f"  Score range:      {min(scores):.1%} - {max(scores):.1%}")
    print(f"  By task type:")
    for task_type, ts in sorted(task_scores.items()):
        print(f"    {task_type}: {sum(ts)/len(ts):.1%} ({len(ts)} tasks)")

    aggregate['models'][model_name] = {
        'num_images': len(results),
        'overall_score': sum(scores)/len(scores),
        'min_score': min(scores),
        'max_score': max(scores),
        'task_scores': {k: sum(v)/len(v) for k, v in task_scores.items()},
        'results': results
    }

# Save aggregate
output_file = f"{eval_results}/aggregate_results_{timestamp}.json"
with open(output_file, 'w') as f:
    json.dump(aggregate, f, indent=2)
print(f"\nAggregate saved to: {output_file}")
PYEOF

export EVAL_RESULTS TIMESTAMP

log ""
log "Logs saved to: $LOG_DIR"
log "Results saved to: $EVAL_RESULTS"
