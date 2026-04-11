#!/bin/sh
#$ -cwd
#$ -j y
#
# SGE job script: evaluate a single model on all tasks.
#
# Usage (submitted by submit_all.sh, or manually):
#   qsub -g tga-okazaki -l node_f=1 -l h_rt=24:00:00 \
#     -v MODEL_ENTRY="Qwen/Qwen2.5-VL-7B-Instruct|vllm_normal|vllm|1" \
#     tsubame/run_model.sh

set -eu

# ============================================================================
# Load config (.env is sourced inside config.sh)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "${SCRIPT_DIR}/config.sh"

cd "${PROJECT_DIR}"

# ============================================================================
# Parse MODEL_ENTRY (passed via qsub -v)
# ============================================================================
if [ -z "${MODEL_ENTRY:-}" ]; then
    echo "ERROR: MODEL_ENTRY is not set." >&2
    echo "Usage: qsub -v MODEL_ENTRY='model_id|group|backend|tp' run_model.sh" >&2
    exit 1
fi

IFS='|' read -r MODEL_ID MODEL_GROUP BACKEND MODEL_TP <<EOF
$MODEL_ENTRY
EOF
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
SAFE_MODEL_ID=$(echo "$MODEL_ID" | tr '/' '_')
FAIL_LOG="${RESULT_DIR}/eval_failures_${SAFE_MODEL_ID}.log"
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
    for task in $TASK_CSV; do
        # Look up metric (requires bash for associative arrays, fall back to eval.sh mapping)
        METRIC=""
        case "$task" in
            japanese-heron-bench)        METRIC="heron-bench" ;;
            ja-vlm-bench-in-the-wild)    METRIC="llm-as-a-judge" ;;
            ja-vg-vqa-500)               METRIC="llm-as-a-judge" ;;
            jmmmu)                       METRIC="jmmmu" ;;
            ja-multi-image-vqa)          METRIC="llm-as-a-judge" ;;
            jdocqa)                      METRIC="llm-as-a-judge" ;;
            mmmu)                        METRIC="mmmu" ;;
            llava-bench-in-the-wild)     METRIC="llm-as-a-judge" ;;
            mecha-ja)                    METRIC="mecha-ja" ;;
            cc-ocr)                      METRIC="cc-ocr" ;;
            ai2d)                        METRIC="ai2d" ;;
            cvqa|docvqa|infographicvqa)  METRIC="substring-match" ;;
            textvqa|chartqa|chartqapro)  METRIC="substring-match" ;;
            okvqa)                       METRIC="substring-match" ;;
            mathvista)                   METRIC="mathvista" ;;
            *)                           METRIC="substring-match" ;;
        esac
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
