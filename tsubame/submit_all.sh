#!/usr/bin/env bash
# Submit SGE jobs for all (or selected) models on TSUBAME 4.0.
#
# Usage:
#   bash tsubame/submit_all.sh                          # Submit all models
#   bash tsubame/submit_all.sh --dry-run                # Show what would be submitted
#   bash tsubame/submit_all.sh --filter vllm            # Only vLLM models
#   bash tsubame/submit_all.sh --filter transformers    # Only transformers models
#   bash tsubame/submit_all.sh --grep "Qwen"            # Only models matching pattern
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ============================================================================
# Parse arguments
# ============================================================================
DRY_RUN=false
BACKEND_FILTER=""
GREP_PATTERN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --filter)    BACKEND_FILTER="$2"; shift 2 ;;
        --grep)      GREP_PATTERN="$2"; shift 2 ;;
        *)           echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ============================================================================
# Validate required settings
# ============================================================================
if [ -z "$TSUBAME_GROUP" ]; then
    echo "ERROR: TSUBAME_GROUP is not set. Add it to .env" >&2
    exit 1
fi

# ============================================================================
# Ensure directories exist
# ============================================================================
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# ============================================================================
# Submit jobs
# ============================================================================
SUBMITTED=0
SKIPPED=0

for entry in "${MODEL_LIST[@]}"; do
    IFS='|' read -r model_id model_group backend model_tp <<< "$entry"

    # Apply filters
    if [ -n "$BACKEND_FILTER" ] && [ "$backend" != "$BACKEND_FILTER" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    if [ -n "$GREP_PATTERN" ] && ! echo "$model_id" | grep -qi "$GREP_PATTERN"; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Sanitize model name for job name and log files
    job_name="eval_$(echo "$model_id" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
    job_name="${job_name:0:100}"

    log_file="${LOG_DIR}/${job_name}.log"

    # Select resource type based on tensor parallel size (GPU count)
    # transformers models don't specify tp; they use 1 GPU
    tp="${model_tp:-1}"
    case "$tp" in
        1) resource="node_q" ;;   # 1 GPU
        2) resource="node_h" ;;   # 2 GPUs
        *) resource="node_f" ;;   # 4 GPUs (full node)
    esac

    # Build qsub arguments (Altair Grid Engine / SGE)
    qsub_args=(
        -g "$TSUBAME_GROUP"
        -N "$job_name"
        -o "$log_file"
        -e "$log_file"
        -l "${resource}=1"
        -l "h_rt=${TSUBAME_H_RT}"
        -v "MODEL_ENTRY=${entry}"
    )

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] qsub ${qsub_args[*]} ${SCRIPT_DIR}/run_model.sh"
    else
        job_id=$(qsub "${qsub_args[@]}" "${SCRIPT_DIR}/run_model.sh")
        echo "Submitted: $model_id -> $job_id"
    fi
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "Done. Submitted: $SUBMITTED, Skipped: $SKIPPED"
if [ "$DRY_RUN" = true ]; then
    echo "(dry-run mode — no jobs were actually submitted)"
fi
