#!/bin/bash
# run_query_benchmark_batch.sh
# Batch runner for:
#   1. Stage 1 spatial graph generation (if missing)
#   2. Query benchmark evaluation across multiple images and models
#
# Run from `cherry/`:
#   chmod +x scripts/run_query_benchmark_batch.sh
#   ./scripts/run_query_benchmark_batch.sh
#
# Environment knobs:
#   IMAGE_FILTER=living_room2
#   IMAGE_LIMIT=1
#   MODEL_FILTER=openai,claude
#   STAGE1_DEVICE=cpu
#
# Examples:
#   IMAGE_FILTER=living_room2 IMAGE_LIMIT=1 ./scripts/run_query_benchmark_batch.sh
#   IMAGE_FILTER=living_room2 IMAGE_LIMIT=1 MODEL_FILTER=openai ./scripts/run_query_benchmark_batch.sh
#   MODEL_FILTER=openai,claude ./scripts/run_query_benchmark_batch.sh
#   MODEL_FILTER='vllm:Qwen/Qwen3-VL-8B-Instruct' ./scripts/run_query_benchmark_batch.sh
#
# Notes:
# - Stage 1 runs once per image and is skipped if the spatial graph already exists.
# - Outputs go under query_benchmark_outputs/<image>__<provider>__<model>/.
# - If Stage 1 needs to run for a new image, OPENAI_API_KEY must be set.
# - On macOS, Stage 1 should usually run with STAGE1_DEVICE=cpu.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHERRY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$CHERRY_DIR"

# ============================================================
# Configuration
# ============================================================

# Images to process.
# Optional environment overrides:
#   IMAGE_FILTER=living_room2
#   IMAGE_LIMIT=1
IMAGE_FILTER="${IMAGE_FILTER:-}"
MODEL_FILTER="${MODEL_FILTER:-}"
# Example:
# IMAGES=(
#   "photos/living_room2.jpg"
#   "photos/kitchen.jpg"
# )
IMAGES=()
while IFS= read -r image_path; do
  IMAGES+=("$image_path")
done < <(find photos -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | sort)

if [ -n "$IMAGE_FILTER" ]; then
  FILTERED=()
  for image_path in "${IMAGES[@]}"; do
    if [[ "$(basename "$image_path")" == *"${IMAGE_FILTER}"* ]]; then
      FILTERED+=("$image_path")
    fi
  done
  IMAGES=("${FILTERED[@]}")
fi

# Models to evaluate.
# Format: "provider|model_name|extra_mode"
# extra_mode:
#   - "default"
#   - "use_vllm"
MODELS=(
  "openai|gpt-4o|default"
  "claude|claude-sonnet-4-5|default"
  # "vllm|Qwen/Qwen3-VL-8B-Instruct|default"
  # "qwen|Qwen/Qwen2.5-VL-7B-Instruct|default"
)

# Benchmark settings
OUTPUT_ROOT="query_benchmark_outputs"
SPATIAL_OUTPUT_DIR="spatial_outputs"
QUERIES_PER_BUCKET=4
SEED=42
MAX_QUERIES=""
IMAGE_LIMIT="${IMAGE_LIMIT:-0}"

if [ "$(uname -s)" = "Darwin" ]; then
  STAGE1_DEVICE="${STAGE1_DEVICE:-cpu}"
else
  STAGE1_DEVICE="${STAGE1_DEVICE:-cuda}"
fi

# vLLM configuration (used only when provider=vllm)
VLLM_URL="http://localhost:8000/v1"
QWEN_READY=false
VLLM_READY=false

# ============================================================
# Helpers
# ============================================================

log() {
  echo "[$(date '+%H:%M:%S')] $1"
}

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "ERROR: Required file not found: $path" >&2
    exit 1
  fi
}

sanitize_name() {
  local value="$1"
  value="${value//[^[:alnum:]_-]/_}"
  while [[ "$value" == *"__"* ]]; do
    value="${value//__/_}"
  done
  value="${value##_}"
  value="${value%%_}"
  if [ -z "$value" ]; then
    value="unknown"
  fi
  printf "%s" "$value"
}

image_stem() {
  local image_path="$1"
  basename "$image_path" | sed 's/\.[^.]*$//'
}

model_output_dir() {
  local image_name="$1"
  local provider="$2"
  local model="$3"
  printf "%s/%s__%s__%s" \
    "$OUTPUT_ROOT" \
    "$(sanitize_name "$image_name")" \
    "$(sanitize_name "$provider")" \
    "$(sanitize_name "$model")"
}

python_module_ready() {
  local module_name="$1"
  python - "$module_name" <<'PY' >/dev/null 2>&1
import importlib
import sys
importlib.import_module(sys.argv[1])
PY
}

check_qwen_ready() {
  python_module_ready torch && \
  python_module_ready transformers && \
  python_module_ready qwen_vl_utils
}

check_vllm_ready() {
  curl -fsS "${VLLM_URL}/models" >/dev/null 2>&1
}

json_list_length() {
  local json_path="$1"
  python - "$json_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
if isinstance(data, list):
    print(len(data))
elif isinstance(data, dict) and "prompts" in data and isinstance(data["prompts"], list):
    print(len(data["prompts"]))
else:
    print(0)
PY
}

stage1_generate_for_image() {
  local image_path="$1"
  local image_name="$2"
  local graph_file="${SPATIAL_OUTPUT_DIR}/${image_name}_spatial_graph.json"

  if [ -f "$graph_file" ]; then
    log "Stage 1 skip for ${image_name}: spatial graph already exists"
    return 0
  fi

  if [ -z "${OPENAI_API_KEY:-}" ]; then
    log "Stage 1 cannot run for ${image_name}: OPENAI_API_KEY is not set"
    return 1
  fi

  local temp_dir="${SPATIAL_OUTPUT_DIR}/temp_${image_name}"
  local detection_output="${temp_dir}/detection_output.json"
  local prompts_file="${temp_dir}/prompts.json"

  mkdir -p "$temp_dir"

  log "Stage 1 object detection for ${image_name}"
  python legacy/generate_queries.py \
    --image "$image_path" \
    --auto-detect-objects \
    --output "$detection_output"

  if [ ! -f "$prompts_file" ] && [ -f "$detection_output" ]; then
    python - "$detection_output" "$prompts_file" <<'PY'
import json
import sys
from pathlib import Path

detection_path = Path(sys.argv[1])
prompts_path = Path(sys.argv[2])
data = json.loads(detection_path.read_text())
prompts = data.get("detected_objects", data.get("prompts", []))
if isinstance(prompts, dict):
    prompts = list(prompts.keys())
prompts_path.write_text(json.dumps(prompts, indent=2))
PY
  fi

  require_file "$prompts_file"
  local prompt_count
  prompt_count="$(json_list_length "$prompts_file")"
  if [ "$prompt_count" -eq 0 ]; then
    log "Stage 1 cannot run for ${image_name}: object detection returned zero prompts"
    rm -rf "$temp_dir"
    return 1
  fi

  log "Stage 1 SAM3 + depth for ${image_name}"
  if ! python depth_sam3_connector.py \
    --image "$image_path" \
    --prompts-file "$prompts_file" \
    --output_dir "$SPATIAL_OUTPUT_DIR" \
    --device "$STAGE1_DEVICE" \
    --no_masks; then
    log "Stage 1 failed for ${image_name}: SAM3 + depth did not complete"
    rm -rf "$temp_dir"
    return 1
  fi

  require_file "$graph_file"
  rm -rf "$temp_dir"
  log "Stage 1 complete for ${image_name}"
}

run_query_eval_for_model() {
  local image_name="$1"
  local graph_file="$2"
  local provider="$3"
  local model="$4"
  local extra_mode="$5"

  local run_dir
  run_dir="$(model_output_dir "$image_name" "$provider" "$model")"
  local eval_file="${run_dir}/evaluation_results.json"

  if [ -f "$eval_file" ]; then
    log "Skip evaluation for ${image_name} / ${provider} / ${model}: results already exist"
    return 0
  fi

  local cmd=(
    python run_query_benchmark.py
    --graph "$graph_file"
    --output-dir "$OUTPUT_ROOT"
    --queries-per-bucket "$QUERIES_PER_BUCKET"
    --seed "$SEED"
    --render-image
    --evaluate
    --provider "$provider"
    --model "$model"
  )

  if [ -n "$MAX_QUERIES" ]; then
    cmd+=(--max-queries "$MAX_QUERIES")
  fi

  if [ "$provider" = "vllm" ]; then
    cmd+=(--vllm-url "$VLLM_URL")
  fi

  if [ "$extra_mode" = "use_vllm" ]; then
    cmd+=(--use-vllm)
  fi

  log "Running query benchmark for ${image_name} / ${provider} / ${model}"
  "${cmd[@]}"

  require_file "$eval_file"

  log "Analyzing results for ${image_name} / ${provider} / ${model}"
  python analyze_query_results.py \
    --input "$eval_file" \
    --output-dir "$run_dir"
}

# ============================================================
# Pre-flight checks
# ============================================================

require_file "run_query_benchmark.py"
require_file "analyze_query_results.py"
require_file "depth_sam3_connector.py"
require_file "legacy/generate_queries.py"

mkdir -p "$OUTPUT_ROOT" "$SPATIAL_OUTPUT_DIR"

if [ -n "$MODEL_FILTER" ]; then
  FILTERED_MODELS=()
  IFS=',' read -r -a MODEL_FILTER_TOKENS <<< "$MODEL_FILTER"
  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r provider model extra_mode <<< "$model_spec"
    for token in "${MODEL_FILTER_TOKENS[@]}"; do
      if [ "$provider" = "$token" ] || [ "${provider}:${model}" = "$token" ]; then
        FILTERED_MODELS+=("$model_spec")
        break
      fi
    done
  done
  MODELS=("${FILTERED_MODELS[@]}")
fi

if [ "${#IMAGES[@]}" -eq 0 ]; then
  echo "ERROR: No images found in photos/" >&2
  exit 1
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "ERROR: No models selected. Check MODEL_FILTER or uncomment entries in MODELS." >&2
  exit 1
fi

if [ "$IMAGE_LIMIT" -gt 0 ] && [ "${#IMAGES[@]}" -gt "$IMAGE_LIMIT" ]; then
  IMAGES=("${IMAGES[@]:0:$IMAGE_LIMIT}")
fi

if check_qwen_ready; then
  QWEN_READY=true
fi

if check_vllm_ready; then
  VLLM_READY=true
fi

log "============================================================"
log "Query Benchmark Batch Run"
log "============================================================"
log "Images: ${#IMAGES[@]}"
log "Models: ${#MODELS[@]}"
log "Queries per bucket: ${QUERIES_PER_BUCKET}"
log "Image limit: ${IMAGE_LIMIT}"
log "Stage 1 device: ${STAGE1_DEVICE}"
if [ -n "$IMAGE_FILTER" ]; then
  log "Image filter: ${IMAGE_FILTER}"
fi
if [ -n "$MODEL_FILTER" ]; then
  log "Model filter: ${MODEL_FILTER}"
fi
if [ -n "$MAX_QUERIES" ]; then
  log "Max queries: ${MAX_QUERIES}"
fi
log "Qwen local ready: ${QWEN_READY}"
log "vLLM server ready: ${VLLM_READY}"
log ""

# ============================================================
# Main loop
# ============================================================

for image_path in "${IMAGES[@]}"; do
  image_name="$(image_stem "$image_path")"
  graph_file="${SPATIAL_OUTPUT_DIR}/${image_name}_spatial_graph.json"

  log "------------------------------------------------------------"
  log "Image: ${image_name}"
  log "------------------------------------------------------------"

  if ! stage1_generate_for_image "$image_path" "$image_name"; then
    log "Skipping ${image_name}: Stage 1 did not complete"
    log ""
    continue
  fi

  require_file "$graph_file"

  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r provider model extra_mode <<< "$model_spec"

    if [ "$provider" = "openai" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
      log "Skip ${image_name} / openai / ${model}: OPENAI_API_KEY not set"
      continue
    fi

    if [ "$provider" = "claude" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
      log "Skip ${image_name} / claude / ${model}: ANTHROPIC_API_KEY not set"
      continue
    fi

    if [ "$provider" = "qwen" ] && [ "$QWEN_READY" != "true" ]; then
      log "Skip ${image_name} / qwen / ${model}: local Qwen dependencies are not available"
      continue
    fi

    if [ "$provider" = "vllm" ] && [ "$VLLM_READY" != "true" ]; then
      log "Skip ${image_name} / vllm / ${model}: vLLM server is not reachable at ${VLLM_URL}"
      continue
    fi

    run_query_eval_for_model "$image_name" "$graph_file" "$provider" "$model" "$extra_mode"
  done

  log ""
done

log "============================================================"
log "Batch run complete"
log "============================================================"
