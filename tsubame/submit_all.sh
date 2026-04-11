#!/usr/bin/env bash
# Submit PBS jobs for all (or selected) models.
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
    # PBS job names: max 236 chars, alphanumeric + underscore/hyphen
    job_name="${job_name:0:100}"

    log_file="${LOG_DIR}/${job_name}.log"

    # Build qsub arguments
    qsub_args=(
        -N "$job_name"
        -v "MODEL_ENTRY=${entry}"
        -o "$log_file"
        -e "$log_file"
        -q "$TSUBAME_QUEUE"
        -l "walltime=${TSUBAME_WALLTIME}"
    )
    [ -n "$TSUBAME_GROUP" ] && qsub_args+=(-W "group_list=${TSUBAME_GROUP}")

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
