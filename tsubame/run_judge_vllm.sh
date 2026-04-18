#!/bin/sh
#$ -cwd
#$ -j y
#
# SGE job script: judge-only scoring across existing prediction.jsonl files,
# using a locally-loaded vLLM judge (default: openai/gpt-oss-20b).
#
# Usage (typical: node_q, 1 GPU is enough for gpt-oss-20b):
#   qsub -g tga-okazaki -l node_q=1 -l h_rt=12:00:00 \
#        tsubame/run_judge_vllm.sh
#
# Tunable via -v (qsub):
#   JUDGE_MODEL        — HF id, default openai/gpt-oss-20b
#   JUDGE_TP           — tensor_parallel_size, default 1
#   JUDGE_MODEL_GROUP  — uv dep group to use, default vllm_normal
#   JUDGE_FLAGS        — extra args forwarded to run_judge_all_vllm.py
#                        e.g. --local_only / --judge_only / --dry_run /
#                        --task_filter japanese-heron-bench / --limit 3
set -eu

# Match run_model.sh: load CUDA 13 so flashinfer GDN kernels resolve
# libcudart.so.13 (needed only for GDN-attention judges, harmless otherwise).
if command -v module >/dev/null 2>&1; then
    module load cuda/13.1.1 2>/dev/null || true
fi

export PATH="$HOME/.local/bin:$PATH"
. ./tsubame/config.sh

cd "${PROJECT_DIR}"

# NOTE: config.sh sets JUDGE_MODEL (for OpenAI-path inference). For the
# local-vLLM judge we use a *separate* variable so we don't clobber it.
JUDGE_VLLM_MODEL="${JUDGE_VLLM_MODEL:-openai/gpt-oss-20b}"
JUDGE_TP="${JUDGE_TP:-1}"
JUDGE_MODEL_GROUP="${JUDGE_MODEL_GROUP:-vllm_normal}"
JUDGE_MODE="${JUDGE_MODE:-full}"          # full | local_only | judge_only
JUDGE_TASK_FILTER="${JUDGE_TASK_FILTER:-}"
JUDGE_MODEL_FILTER="${JUDGE_MODEL_FILTER:-}"
JUDGE_LIMIT="${JUDGE_LIMIT:-0}"
JUDGE_OVERWRITE="${JUDGE_OVERWRITE:-0}"
JUDGE_SHARD="${JUDGE_SHARD:-}"

EXTRA_FLAGS=""
case "$JUDGE_MODE" in
    local_only) EXTRA_FLAGS="$EXTRA_FLAGS --local_only" ;;
    judge_only) EXTRA_FLAGS="$EXTRA_FLAGS --judge_only" ;;
    full|"") ;;
    *) echo "ERROR: unknown JUDGE_MODE=$JUDGE_MODE" >&2; exit 1 ;;
esac
[ -n "$JUDGE_TASK_FILTER" ]  && EXTRA_FLAGS="$EXTRA_FLAGS --task_filter $JUDGE_TASK_FILTER"
[ -n "$JUDGE_MODEL_FILTER" ] && EXTRA_FLAGS="$EXTRA_FLAGS --model_filter $JUDGE_MODEL_FILTER"
[ "$JUDGE_LIMIT" != "0" ]    && EXTRA_FLAGS="$EXTRA_FLAGS --limit $JUDGE_LIMIT"
[ "$JUDGE_OVERWRITE" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --overwrite"
[ -n "$JUDGE_SHARD" ]        && EXTRA_FLAGS="$EXTRA_FLAGS --model_shard $JUDGE_SHARD"

echo "========================================"
echo "Judge model : $JUDGE_VLLM_MODEL (tp=$JUDGE_TP)"
echo "Group       : $JUDGE_MODEL_GROUP"
echo "Mode        : $JUDGE_MODE"
echo "Result dir  : $RESULT_DIR"
echo "Extra flags : $EXTRA_FLAGS"
echo "Host        : $(hostname)"
echo "GPUs        : $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Start       : $(date -Iseconds)"
echo "========================================"

export UV_PROJECT_ENVIRONMENT="${PROJECT_DIR}/.venvs/${JUDGE_MODEL_GROUP}"
uv sync --group "$JUDGE_MODEL_GROUP"

# shellcheck disable=SC2086
uv run --group "$JUDGE_MODEL_GROUP" python scripts/run_judge_all_vllm.py \
    --result_dir "$RESULT_DIR" \
    --judge_model "$JUDGE_VLLM_MODEL" \
    --tensor_parallel_size "$JUDGE_TP" \
    $EXTRA_FLAGS

echo "========================================"
echo "End : $(date -Iseconds)"
echo "========================================"
