#!/usr/bin/env bash
# Submit the vLLM judge job to SGE (TSUBAME 4.0).
# Mirrors the submit_all.sh conventions: sources config.sh to pick up
# TSUBAME_GROUP/LOG_DIR/RESULT_DIR from .env.
#
# Usage examples:
#   bash tsubame/submit_judge_vllm.sh --smoke                        # 2 cells, 1h
#   bash tsubame/submit_judge_vllm.sh                                # full run
#   bash tsubame/submit_judge_vllm.sh --local-only                   # no LLM
#   bash tsubame/submit_judge_vllm.sh --task-filter ai2d --limit 3   # custom
#   bash tsubame/submit_judge_vllm.sh --dry-run                      # print only
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config.sh"

H_RT="${TSUBAME_H_RT:-12:00:00}"
RESOURCE="node_q"
JOB_NAME="judge_vllm"
MODE="full"
TASK_FILTER=""
MODEL_FILTER=""
LIMIT=0
OVERWRITE=0
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            H_RT="1:00:00"
            TASK_FILTER="japanese-heron-bench"
            LIMIT=2
            JOB_NAME="judge_smoke"
            shift ;;
        --local-only)   MODE="local_only"; JOB_NAME="judge_local"; shift ;;
        --judge-only)   MODE="judge_only"; JOB_NAME="judge_only"; shift ;;
        --task-filter)  TASK_FILTER="$2"; shift 2 ;;
        --model-filter) MODEL_FILTER="$2"; shift 2 ;;
        --limit)        LIMIT="$2"; shift 2 ;;
        --overwrite)    OVERWRITE=1; shift ;;
        --h-rt)         H_RT="$2"; shift 2 ;;
        --resource)     RESOURCE="$2"; shift 2 ;;
        --name)         JOB_NAME="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ -z "${TSUBAME_GROUP:-}" ]; then
    echo "ERROR: TSUBAME_GROUP is not set in .env" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$RESULT_DIR"
LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

# Pass each option as a separate env var (SGE -v does not handle spaces).
VARS="JUDGE_MODE=${MODE}"
[ -n "$TASK_FILTER" ]  && VARS="${VARS},JUDGE_TASK_FILTER=${TASK_FILTER}"
[ -n "$MODEL_FILTER" ] && VARS="${VARS},JUDGE_MODEL_FILTER=${MODEL_FILTER}"
[ "$LIMIT" != "0" ]    && VARS="${VARS},JUDGE_LIMIT=${LIMIT}"
[ "$OVERWRITE" = "1" ] && VARS="${VARS},JUDGE_OVERWRITE=1"

qsub_args=(
    -g "$TSUBAME_GROUP"
    -N "$JOB_NAME"
    -o "$LOG_FILE"
    -e "$LOG_FILE"
    -l "${RESOURCE}=1"
    -l "h_rt=${H_RT}"
    -v "$VARS"
)

echo "qsub ${qsub_args[*]} ${SCRIPT_DIR}/run_judge_vllm.sh"
echo "Log: ${LOG_FILE}"

if [ "$DRY_RUN" = true ]; then
    echo "(dry-run; not submitting)"
    exit 0
fi

qsub "${qsub_args[@]}" "${SCRIPT_DIR}/run_judge_vllm.sh"
