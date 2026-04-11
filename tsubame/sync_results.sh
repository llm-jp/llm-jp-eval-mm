#!/usr/bin/env bash
# Sync TSUBAME evaluation results to local result/ directory.
# Run this on the mdx side so the Runner dashboard picks up TSUBAME results.
#
# Usage:
#   bash tsubame/sync_results.sh              # 1回だけ同期
#   bash tsubame/sync_results.sh --watch      # 5分ごとに同期（Ctrl+C で停止）
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

# TSUBAME の結果ディレクトリ（SSH 経由）
TSUBAME_HOST="${TSUBAME_HOST:-tsubame}"
TSUBAME_RESULT_DIR="${TSUBAME_RESULT_DIR:-/gs/bs/tga-okazaki/maeda/eval-mm-results}"

# ローカルの結果ディレクトリ
LOCAL_RESULT_DIR="${PROJECT_ROOT}/result"

INTERVAL="${SYNC_INTERVAL:-300}"  # デフォルト5分
WATCH=false

for arg in "$@"; do
    case "$arg" in
        --watch) WATCH=true ;;
    esac
done

do_sync() {
    echo "[$(date '+%H:%M:%S')] Syncing from ${TSUBAME_HOST}:${TSUBAME_RESULT_DIR}/ ..."
    rsync -avz --ignore-existing \
        "${TSUBAME_HOST}:${TSUBAME_RESULT_DIR}/" \
        "${LOCAL_RESULT_DIR}/"
    echo "[$(date '+%H:%M:%S')] Done."
}

if [ "$WATCH" = true ]; then
    echo "Syncing every $((INTERVAL / 60)) minutes. Ctrl+C to stop."
    while true; do
        do_sync
        sleep "$INTERVAL"
    done
else
    do_sync
fi
