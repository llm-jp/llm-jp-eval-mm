#!/usr/bin/env bash
# Monitor TSUBAME evaluation job status.
#
# Usage:
#   bash tsubame/monitor.sh            # 1回表示
#   bash tsubame/monitor.sh --watch    # 30秒ごとに更新
#   bash tsubame/monitor.sh --logs     # 最新ログの末尾も表示
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

SHOW_LOGS=false
WATCH=false

for arg in "$@"; do
    case "$arg" in
        --logs)  SHOW_LOGS=true ;;
        --watch) WATCH=true ;;
    esac
done

show_status() {
    echo "========================================"
    echo "  eval-mm TSUBAME Monitor  $(date '+%H:%M:%S')"
    echo "========================================"

    # --- Running jobs ---
    echo ""
    echo "## Jobs (qstat)"
    QSTAT_OUT=$(qstat 2>/dev/null || true)
    if [ -n "$QSTAT_OUT" ]; then
        echo "$QSTAT_OUT"
    else
        echo "  (no running jobs)"
    fi

    # --- Results ---
    echo ""
    echo "## Results"
    if [ -d "$RESULT_DIR" ]; then
        TOTAL_PREDICTIONS=$(find "$RESULT_DIR" -name "prediction.jsonl" 2>/dev/null | wc -l)
        TOTAL_MODELS=$(find "$RESULT_DIR" -mindepth 2 -maxdepth 3 -name "prediction.jsonl" 2>/dev/null \
            | xargs -I{} dirname {} | xargs -I{} dirname {} | sort -u | wc -l)
        echo "  Predictions: $TOTAL_PREDICTIONS (across $TOTAL_MODELS models)"
    else
        echo "  (no results yet)"
    fi

    # --- Failures ---
    echo ""
    echo "## Failures"
    FAIL_COUNT=0
    if [ -d "$RESULT_DIR" ]; then
        for f in "$RESULT_DIR"/eval_failures_*.log; do
            [ -f "$f" ] || continue
            if [ -s "$f" ]; then
                FAIL_COUNT=$((FAIL_COUNT + $(wc -l < "$f")))
                echo "  $(basename "$f"):"
                sed 's/^/    /' "$f"
            fi
        done
    fi
    if [ "$FAIL_COUNT" -eq 0 ]; then
        echo "  (none)"
    fi

    # --- Recent logs ---
    if [ "$SHOW_LOGS" = true ] && [ -d "$LOG_DIR" ]; then
        echo ""
        echo "## Latest log"
        LATEST=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "  $LATEST"
            echo "  ---"
            tail -20 "$LATEST" | sed 's/^/  /'
        else
            echo "  (no logs yet)"
        fi
    fi

    echo ""
    echo "========================================"
}

if [ "$WATCH" = true ]; then
    while true; do
        clear
        show_status
        sleep 30
    done
else
    show_status
fi
