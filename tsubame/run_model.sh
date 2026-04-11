#!/usr/bin/env bash
# PBS job script: evaluate a single model on all tasks.
#
# Usage (submitted by submit_all.sh, or manually):
#   qsub -v MODEL_ENTRY="Qwen/Qwen2.5-VL-7B-Instruct|vllm_normal|vllm|1" tsubame/run_model.sh
#
# PBS directives are defaults; submit_all.sh overrides -N, -o, -e per model.
#PBS -q gpu_h100
#PBS -l select=1:ngpus=4:ncpus=16:mem=200gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -eu

# ============================================================================
# Resolve paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

cd "${PROJECT_DIR}"

# Load .env if present (HF_TOKEN, OPENAI_API_KEY, etc.)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ============================================================================
# Parse MODEL_ENTRY (passed via qsub -v)
# ============================================================================
if [ -z "${MODEL_ENTRY:-}" ]; then
    echo "ERROR: MODEL_ENTRY is not set." >&2
    echo "Usage: qsub -v MODEL_ENTRY='model_id|group|backend|tp' run_model.sh" >&2
    exit 1
fi

IFS='|' read -r MODEL_ID MODEL_GROUP BACKEND MODEL_TP <<< "$MODEL_ENTRY"
TP="${MODEL_TP:-$TENSOR_PARALLEL_SIZE}"

echo "========================================"
echo "Model:   $MODEL_ID"
echo "Group:   $MODEL_GROUP"
echo "Backend: $BACKEND"
echo "TP:      $TP"
echo "Host:    $(hostname)"
echo "GPUs:    $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Start:   $(date -Iseconds)"
echo "========================================"

# ============================================================================
# Sync dependencies for this model's group
# ============================================================================
echo ">>> Syncing group: $MODEL_GROUP"
uv sync --group "$MODEL_GROUP"

# ============================================================================
# Run evaluation
# ============================================================================
mkdir -p "$RESULT_DIR"
FAIL_LOG="${RESULT_DIR}/eval_failures_${MODEL_ID//\//_}.log"
> "$FAIL_LOG"

if [ "$BACKEND" = "vllm" ]; then
    # vLLM: load model once, run ALL tasks
    echo ">>> [ALL TASKS] $MODEL_ID (vllm, tp=$TP)"
    uv run --group "$MODEL_GROUP" python examples/sample_vllm_multi.py \
        --model_id "$MODEL_ID" \
        --task_ids "$TASK_CSV" \
        --metrics "$METRIC_CSV" \
        --judge_model "$JUDGE_MODEL" \
        --result_dir "$RESULT_DIR" \
        --tensor_parallel_size "$TP" \
        --inference_only \
        --batch_chunk_size 50 \
    || echo "WARN: $MODEL_ID had errors (see $FAIL_LOG)"
else
    # Transformers: per-task loop
    for task in "${TASK_LIST[@]}"; do
        METRIC="${METRIC_MAP[$task]}"
        echo ">>> [$task] $MODEL_ID (transformers)"
        uv run --group "$MODEL_GROUP" python examples/sample.py \
            --model_id "$MODEL_ID" \
            --task_id "$task" \
            --metrics "$METRIC" \
            --judge_model "$JUDGE_MODEL" \
            --result_dir "$RESULT_DIR" \
            --inference_only \
        || echo "FAIL|$task|$MODEL_ID|$BACKEND" | tee -a "$FAIL_LOG"
    done
fi

# ============================================================================
# Summary
# ============================================================================
echo "========================================"
echo "End:     $(date -Iseconds)"
if [ -s "$FAIL_LOG" ]; then
    echo "Failures:"
    cat "$FAIL_LOG"
else
    echo "All tasks completed successfully."
fi
echo "========================================"
