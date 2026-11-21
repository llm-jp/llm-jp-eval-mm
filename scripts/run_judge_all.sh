#!/usr/bin/env bash
# Run judge-only scoring over existing prediction.jsonl files.
#
# Scans RESULT_DIR/<task>/<org>/<model>/prediction.jsonl and, for each pair
# that does not yet have evaluation.jsonl, invokes `python -m eval_mm evaluate`.
#
# Usage:
#   bash scripts/run_judge_all.sh                       # dry-run (shows plan)
#   RUN=1 bash scripts/run_judge_all.sh                 # actually run
#   RESULT_DIR=/path/to/results RUN=1 bash scripts/run_judge_all.sh
#   MODEL_FILTER="Qwen/" bash scripts/run_judge_all.sh  # only matching models
#   TASK_FILTER="ai2d,mmmu" bash scripts/run_judge_all.sh
#   OVERWRITE=1 RUN=1 bash scripts/run_judge_all.sh     # re-score even if evaluation.jsonl exists
set -eu
cd "$(dirname "$0")/.."

# --- Load .env if present ---
if [ -f .env ]; then set -a; source .env; set +a; fi

RESULT_DIR="${RESULT_DIR:-/gs/bs/tga-okazaki/maeda/eval-mm-results}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.1-2025-11-13}"
BATCH_SIZE="${BATCH_SIZE:-10}"
RUN="${RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
MODEL_FILTER="${MODEL_FILTER:-}"
TASK_FILTER="${TASK_FILTER:-}"
SKIP_TASKS="${SKIP_TASKS:-ja-multi-image-vqa,jdocqa,jmmmu,mmmu,mathvista}"

# task_id -> metric (mirrors tsubame/run_model.sh)
metric_for_task() {
    case "$1" in
        japanese-heron-bench)        echo "heron-bench" ;;
        ja-vlm-bench-in-the-wild)    echo "llm-as-a-judge" ;;
        ja-vg-vqa-500)               echo "llm-as-a-judge" ;;
        jmmmu)                       echo "jmmmu" ;;
        ja-multi-image-vqa)          echo "llm-as-a-judge" ;;
        jdocqa)                      echo "llm-as-a-judge" ;;
        mmmu)                        echo "mmmu" ;;
        llava-bench-in-the-wild)     echo "llm-as-a-judge" ;;
        mecha-ja)                    echo "mecha-ja" ;;
        cc-ocr)                      echo "cc-ocr" ;;
        ai2d)                        echo "ai2d" ;;
        cvqa|docvqa|infographicvqa)  echo "substring-match" ;;
        textvqa|chartqa|chartqapro)  echo "substring-match" ;;
        okvqa)                       echo "substring-match" ;;
        jawildtext-board-vqa)        echo "jawildtext-board-vqa" ;;
        jawildtext-handwriting-ocr)  echo "jawildtext-handwriting-ocr" ;;
        jawildtext-receipt-kie)      echo "jawildtext-receipt-kie" ;;
        mathvista)                   echo "mathvista" ;;
        *)                           echo "substring-match" ;;
    esac
}

in_csv() {
    local needle="$1" csv="$2"
    [ -z "$csv" ] && return 1
    case ",$csv," in (*,"$needle",*) return 0 ;; esac
    return 1
}

total=0; planned=0; skipped_nopred=0; skipped_done=0; skipped_filter=0
FAIL_LOG="${RESULT_DIR}/judge_failures.log"

for task_dir in "$RESULT_DIR"/*/; do
    task=$(basename "$task_dir")
    case "$task" in eval_failures_*|judge_failures*) continue ;; esac
    if in_csv "$task" "$SKIP_TASKS"; then continue; fi
    if [ -n "$TASK_FILTER" ] && ! in_csv "$task" "$TASK_FILTER"; then continue; fi

    metric=$(metric_for_task "$task")

    for org_dir in "$task_dir"*/; do
        [ -d "$org_dir" ] || continue
        for model_dir in "$org_dir"*/; do
            [ -d "$model_dir" ] || continue
            org=$(basename "$org_dir")
            mname=$(basename "$model_dir")
            model_id="${org}/${mname}"
            total=$((total+1))

            if [ -n "$MODEL_FILTER" ] && ! echo "$model_id" | grep -qE "$MODEL_FILTER"; then
                skipped_filter=$((skipped_filter+1)); continue
            fi
            if [ ! -f "${model_dir}prediction.jsonl" ]; then
                skipped_nopred=$((skipped_nopred+1)); continue
            fi
            if [ -f "${model_dir}evaluation.jsonl" ] && [ "$OVERWRITE" != "1" ]; then
                skipped_done=$((skipped_done+1)); continue
            fi

            planned=$((planned+1))
            echo "[$planned] $task :: $model_id  (metric=$metric)"
            if [ "$RUN" = "1" ]; then
                uv run python -m eval_mm evaluate \
                    --model_id "$model_id" \
                    --task_id "$task" \
                    --metrics "$metric" \
                    --judge_model "$JUDGE_MODEL" \
                    --batch_size_for_evaluation "$BATCH_SIZE" \
                    --result_dir "$RESULT_DIR" \
                || echo "FAIL|$task|$model_id" | tee -a "$FAIL_LOG"
            fi
        done
    done
done

echo "========================================"
echo "Total cells scanned   : $total"
echo "Planned judge runs    : $planned"
echo "Skipped (already done): $skipped_done"
echo "Skipped (no pred)     : $skipped_nopred"
echo "Skipped (by filter)   : $skipped_filter"
echo "Skipped tasks         : $SKIP_TASKS"
if [ "$RUN" != "1" ]; then
    echo "(dry-run; set RUN=1 to actually execute)"
fi
