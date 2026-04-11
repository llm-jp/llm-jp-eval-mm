#!/usr/bin/env bash
# Unified evaluation script for all models.
# Models are ordered from smallest to largest.
# Automatically selects the correct backend (vLLM or transformers) per model.
set -eu

# Ensure we run from the repository root (eval.sh uses relative paths)
cd "$(dirname "$0")"
# Note: -x removed to reduce log noise; -e kept but individual model runs
# are wrapped with || to allow skipping failures.

# Load environment variables (.env) for HF_TOKEN, API keys, etc.
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Ensure HF_HOME is set (prefer /model mount, fallback to workspace)
export HF_HOME="${HF_HOME:-/model/silviase/cache/huggingface}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
TENSOR_PARALLEL_SIZE=4
JUDGE_MODEL="gpt-5.1-2025-11-13"
RESULT_DIR="result"

# Optional filter: only run models matching this backend ("vllm" or "transformers").
# Empty = run all.  Usage: EVAL_BACKEND_FILTER=vllm bash eval.sh
BACKEND_FILTER="${EVAL_BACKEND_FILTER:-}"

# ============================================================================
# Model list (ordered by parameter count, smallest first)
# Format: "model_id|group|backend|tp"
#   backend = "vllm"         -> sample_vllm.py (vllm_registry.py)
#   backend = "transformers"  -> sample.py     (model_table.py)
#   tp      = tensor parallel size (optional, defaults to TENSOR_PARALLEL_SIZE)
#             Small models need tp=1 (attention heads not divisible by 4)
# ============================================================================
declare -a MODEL_LIST=(
    # ~1B (tp=1: small models, attention heads not divisible by 4)
    "OpenGVLab/InternVL3-1B|vllm_normal|vllm|1"
    "OpenGVLab/InternVL3_5-1B|vllm_normal|vllm|1"
    "AIDC-AI/Ovis2-1B|vllm_normal|vllm|1"
    "turing-motors/Heron-NVILA-Lite-1B|heron_nvila|transformers"
    # ~2B (tp=1)
    "OpenGVLab/InternVL3-2B|vllm_normal|vllm|1"
    "OpenGVLab/InternVL3_5-2B|vllm_normal|vllm|1"
    "Qwen/Qwen2-VL-2B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3-VL-2B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3.5-2B|vllm_normal|vllm|1"
    "AIDC-AI/Ovis2-2B|vllm_normal|vllm|1"
    "AIDC-AI/Ovis2.5-2B|vllm_normal|vllm|1"
    "google/gemma-4-E2B-it|gemma4|transformers"
    "turing-motors/Heron-NVILA-Lite-2B|heron_nvila|transformers"
    # ~3-4B (tp=1: fits on single A100-40GB)
    "Qwen/Qwen2.5-VL-3B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3-VL-4B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3.5-4B|vllm_normal|vllm|1"
    "OpenGVLab/InternVL3_5-4B|vllm_normal|vllm|1"
    "google/gemma-3-4b-it|vllm_normal|vllm|1"
    "google/gemma-4-E4B-it|gemma4|transformers"
    "allenai/Molmo2-4B|vllm_normal|vllm|1"
    "AIDC-AI/Ovis2-4B|vllm_normal|vllm|1"
    "moonshotai/Kimi-VL-A3B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3-VL-30B-A3B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3.5-35B-A3B|vllm_normal|vllm|1"
    "google/gemma-4-26B-A4B-it|gemma4|transformers"
    # ~7-9B (tp=1: fits on single A100-40GB)
    "llava-hf/llava-1.5-7b-hf|vllm_normal|vllm|1"
    "llava-hf/llava-v1.6-mistral-7b-hf|vllm_normal|vllm|1"
    "neulab/Pangea-7B-hf|vllm_normal|vllm|1"
    "Qwen/Qwen2-VL-7B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen2.5-VL-7B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3-VL-8B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3.5-9B|vllm_normal|vllm|1"
    "OpenGVLab/InternVL2-8B|vllm_normal|vllm|1"
    "OpenGVLab/InternVL3-8B|vllm_normal|vllm|1"
    "OpenGVLab/InternVL3_5-8B|vllm_normal|vllm|1"
    "CohereLabs/aya-vision-8b|vllm_normal|vllm|1"
    "allenai/Molmo2-8B|vllm_normal|vllm|1"
    "AIDC-AI/Ovis2-8B|vllm_normal|vllm|1"
    "AIDC-AI/Ovis2.5-9B|vllm_normal|vllm|1"
    "sbintuitions/sarashina2-vision-8b|sarashina|transformers"
    "SakanaAI/Llama-3-EvoVLM-JP-v2|evovlm|transformers"
    "openbmb/MiniCPM-o-2_6|vllm_normal|vllm|1"
    # ~11-15B (tp=1: still fits on A100-40GB in bf16)
    "meta-llama/Llama-3.2-11B-Vision-Instruct|normal|transformers"
    "mistralai/Pixtral-12B-2409|vllm_normal|transformers"
    "google/gemma-3-12b-it|vllm_normal|vllm|1"
    "llava-hf/llava-1.5-13b-hf|vllm_normal|vllm|1"
    "OpenGVLab/InternVL3-14B|vllm_normal|vllm|1"
    "MIL-UT/Asagi-14B|old|transformers"
    "sbintuitions/sarashina2-vision-14b|sarashina|transformers"
    # "llm-jp/llm-jp-3-vila-14b|vilaja|transformers"  # SKIP: vilaja group broken (No module named 'vila.constants')
    "microsoft/Phi-4-multimodal-instruct|vllm_normal|vllm|1"
    "turing-motors/Heron-NVILA-Lite-15B|heron_nvila|transformers"
    "AIDC-AI/Ovis2-16B|vllm_normal|vllm|2"
    # ~24-34B (tp=2 or tp=4)
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503|vllm_normal|transformers"
    "OpenGVLab/InternVL2-26B|vllm_normal|vllm|2"
    "google/gemma-3-27b-it|vllm_normal|vllm|2"
    "Qwen/Qwen3.5-27B|vllm_normal|vllm|2"
    "Qwen/Qwen3-VL-32B-Instruct|vllm_normal|vllm|2"
    "Qwen/Qwen2.5-VL-32B-Instruct|vllm_normal|vllm|2"
    "CohereLabs/aya-vision-32b|vllm_normal|vllm|2"
    "google/gemma-4-31B-it|gemma4|transformers"
    "turing-motors/Heron-NVILA-Lite-33B|heron_nvila|transformers"
    "AIDC-AI/Ovis2-34B|vllm_normal|vllm|2"
    "OpenGVLab/InternVL3-38B|vllm_normal|vllm|2"
    "OpenGVLab/InternVL3_5-38B|vllm_normal|vllm|2"
    # ~72B+ (tp=4: need all GPUs)
    "Qwen/Qwen2-VL-72B-Instruct|vllm_normal|vllm|4"
    "Qwen/Qwen2.5-VL-72B-Instruct|vllm_normal|vllm|4"
    "OpenGVLab/InternVL3-78B|vllm_normal|vllm|4"
    "meta-llama/Llama-3.2-90B-Vision-Instruct|normal|transformers"
    "deepseek-ai/deepseek-vl2|vllm_normal|vllm|4"
    # Size unknown / special
    "zai-org/GLM-4.5V|vllm_normal|vllm"
    "zai-org/GLM-4.6V|vllm_normal|vllm"
    "zai-org/GLM-4.6V-Flash|vllm_normal|vllm"
    "cyberagent/llava-calm2-siglip|calm|transformers"
    "stabilityai/japanese-instructblip-alpha|stablevlm|transformers"
    "stabilityai/japanese-stable-vlm|stablevlm|transformers"
    # API-based
    "gpt-4o-2024-11-20|normal|transformers"
)

# ============================================================================
# Tasks & Metrics
# ============================================================================
declare -a task_list=(
    "japanese-heron-bench"
    "ja-vlm-bench-in-the-wild"
    "ja-vg-vqa-500"
    "jmmmu"
    "ja-multi-image-vqa"
    "jdocqa"
    "mmmu"
    "llava-bench-in-the-wild"
    # "jic-vqa"  # SKIP: requires scripts/prepare_jic_vqa.py to be run first
    "cvqa"
    "cc-ocr"
    "mecha-ja"
    "ai2d"
    # "blink"  # SKIP: dataset features alignment error (choices field type mismatch)
    "docvqa"
    "infographicvqa"
    "textvqa"
    "chartqa"
    "chartqapro"
    "mathvista"
    "okvqa"
)

declare -A METRIC_MAP=(
    ["japanese-heron-bench"]="heron-bench"
    ["ja-vlm-bench-in-the-wild"]="llm-as-a-judge"
    ["ja-vg-vqa-500"]="llm-as-a-judge"
    ["jmmmu"]="jmmmu"
    ["ja-multi-image-vqa"]="llm-as-a-judge"
    ["jdocqa"]="llm-as-a-judge"
    ["mmmu"]="mmmu"
    ["llava-bench-in-the-wild"]="llm-as-a-judge"
    # ["jic-vqa"]="jic-vqa"  # SKIP
    ["mecha-ja"]="mecha-ja"
    ["cc-ocr"]="cc-ocr"
    ["ai2d"]="ai2d"
    # ["blink"]="blink"  # SKIP
    ["cvqa"]="substring-match"
    ["docvqa"]="substring-match"
    ["infographicvqa"]="substring-match"
    ["textvqa"]="substring-match"
    ["chartqa"]="substring-match"
    ["chartqapro"]="substring-match"
    ["mathvista"]="mathvista"
    ["okvqa"]="substring-match"
)

# ============================================================================
# Main evaluation loop — MODEL-FIRST order
#
# For vLLM models: load the model once, run ALL tasks, then unload.
# For transformers models: run each task individually (per-task process).
# This maximises GPU utilisation by avoiding repeated model load/unload.
# ============================================================================
FAIL_LOG="${RESULT_DIR}/eval_failures.log"
STATUS_FILE="${RESULT_DIR}/.eval_status.json"
mkdir -p "$RESULT_DIR"
> "$FAIL_LOG"

NUM_TASKS=${#task_list[@]}
NUM_MODELS=${#MODEL_LIST[@]}

# When filtering by backend, count only matching models for accurate progress.
if [ -n "$BACKEND_FILTER" ]; then
    FILTERED_MODELS=0
    for _entry in "${MODEL_LIST[@]}"; do
        IFS='|' read -r _ _ _be _ <<< "$_entry"
        [ "$_be" = "$BACKEND_FILTER" ] && FILTERED_MODELS=$((FILTERED_MODELS + 1))
    done
    TOTAL_RUNS=$((NUM_TASKS * FILTERED_MODELS))
    echo "Backend filter: $BACKEND_FILTER ($FILTERED_MODELS models × $NUM_TASKS tasks = $TOTAL_RUNS runs)"
else
    TOTAL_RUNS=$((NUM_TASKS * NUM_MODELS))
fi
COMPLETED=0
FAILED=0
START_EPOCH=$(date +%s)

write_status() {
    local status="$1"
    local now=$(date +%s)
    local elapsed=$((now - START_EPOCH))
    local eta=0
    if [ "$COMPLETED" -gt 0 ]; then
        eta=$(( elapsed * (TOTAL_RUNS - COMPLETED) / COMPLETED ))
    fi
    cat > "$STATUS_FILE" <<EOJSON
{
  "running": $([ "$status" = "running" ] && echo "true" || echo "false"),
  "currentTask": "${CURRENT_TASK:-}",
  "currentModel": "${CURRENT_MODEL:-}",
  "backend": "${CURRENT_BACKEND:-}",
  "completed": $COMPLETED,
  "failed": $FAILED,
  "total": $TOTAL_RUNS,
  "progress": $(( COMPLETED * 100 / TOTAL_RUNS )),
  "etaSeconds": $eta,
  "elapsedSeconds": $elapsed
}
EOJSON
}

# Build comma-separated task and metric lists for multi-task script
TASK_CSV=$(IFS=,; echo "${task_list[*]}")
METRIC_CSV=""
for _t in "${task_list[@]}"; do
    METRIC_CSV="${METRIC_CSV:+$METRIC_CSV,}${METRIC_MAP[$_t]}"
done

write_status "running"
LAST_GROUP=""

for entry in "${MODEL_LIST[@]}"; do
    IFS='|' read -r model_name model_group backend model_tp <<< "$entry"

    # Skip models that don't match the backend filter.
    if [ -n "$BACKEND_FILTER" ] && [ "$backend" != "$BACKEND_FILTER" ]; then
        continue
    fi

    TP="${model_tp:-$TENSOR_PARALLEL_SIZE}"
    CURRENT_MODEL="$model_name"
    CURRENT_BACKEND="$backend"

    # Sync dependencies only when group changes
    if [ "$model_group" != "$LAST_GROUP" ]; then
        echo ">>> Syncing group: $model_group"
        uv sync --group "$model_group"
        LAST_GROUP="$model_group"
    fi

    if [ "$backend" = "vllm" ]; then
        # ── vLLM: load model once, run ALL tasks ────────────────
        echo ">>> [ALL TASKS] $model_name ($model_group/vllm, tp=$TP)"
        CURRENT_TASK="${task_list[0]}"
        write_status "running"

        uv run --group "$model_group" python examples/sample_vllm_multi.py \
            --model_id "$model_name" \
            --task_ids "$TASK_CSV" \
            --metrics "$METRIC_CSV" \
            --judge_model "$JUDGE_MODEL" \
            --result_dir "$RESULT_DIR" \
            --tensor_parallel_size "$TP" \
            --inference_only \
            --batch_chunk_size 50 \
            --status_file "$STATUS_FILE" \
            --fail_log "$FAIL_LOG" \
            --completed_offset "$COMPLETED" \
            --failed_offset "$FAILED" \
            --total_runs "$TOTAL_RUNS" \
            --start_epoch "$START_EPOCH" \
        || echo "WARN: $model_name had errors (see $FAIL_LOG)"

        # Read back completion counts written by the Python script
        if [ -f "$STATUS_FILE" ]; then
            COMPLETED=$(python3 -c "import json; print(json.load(open('$STATUS_FILE'))['completed'])")
            FAILED=$(python3 -c "import json; print(json.load(open('$STATUS_FILE'))['failed'])")
        fi
    else
        # ── Transformers: per-task loop ─────────────────────────
        for task in "${task_list[@]}"; do
            METRIC=${METRIC_MAP[$task]}
            CURRENT_TASK="$task"
            write_status "running"

            echo ">>> [$task] $model_name ($model_group/transformers)"
            uv run --group "$model_group" python examples/sample.py \
                --model_id "$model_name" \
                --task_id "$task" \
                --metrics "$METRIC" \
                --judge_model "$JUDGE_MODEL" \
                --result_dir "$RESULT_DIR" \
                --inference_only \
            || { echo "FAIL|$task|$model_name|$backend" | tee -a "$FAIL_LOG"; FAILED=$((FAILED + 1)); }
            COMPLETED=$((COMPLETED + 1))
        done
    fi
done

write_status "done"

echo ""
echo "All evaluations are done."
if [ -s "$FAIL_LOG" ]; then
    echo "Failures ($(wc -l < "$FAIL_LOG")):"
    cat "$FAIL_LOG"
fi
