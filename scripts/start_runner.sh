#!/usr/bin/env bash
# Start the runner monitoring stack: FastAPI backend + Next.js frontend.
#
# IMPORTANT — Virtual environment isolation strategy:
#   eval.sh runs `uv sync --group <model_group>` which swaps model-specific
#   packages (torch, transformers, vllm, …) in the shared .venv.
#   The API server only needs base dependencies (fastapi, uvicorn) which are
#   NOT in any conflict group and are never removed.
#   By launching the API via `.venv/bin/python` directly (not `uv run`),
#   the running process keeps its loaded modules in memory and is unaffected
#   by later `uv sync` operations.
#
# Usage:
#   bash scripts/start_runner.sh          # start API + web
#   # then in another terminal:
#   EVAL_BACKEND_FILTER=vllm bash eval.sh # run vLLM models only
set -eu
cd "$(dirname "$0")/.."

# ── 1. Ensure base dependencies are installed ───────────────────
echo "[runner] Syncing base dependencies..."
uv sync

# ── 2. Start FastAPI backend ────────────────────────────────────
API_PORT="${API_PORT:-8000}"
echo "[runner] Starting API server on port ${API_PORT}..."
.venv/bin/python -m uvicorn eval_mm.api:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --log-level info &
API_PID=$!

# ── 3. Start Next.js dev server ─────────────────────────────────
WEB_PORT="${WEB_PORT:-3000}"
echo "[runner] Starting web frontend on port ${WEB_PORT}..."
cd web
pnpm install --frozen-lockfile 2>/dev/null || pnpm install
NEXT_PUBLIC_API_URL="http://localhost:${API_PORT}" PORT="$WEB_PORT" pnpm dev &
WEB_PID=$!
cd ..

# ── 4. Cleanup on exit ──────────────────────────────────────────
cleanup() {
    echo ""
    echo "[runner] Shutting down..."
    kill "$API_PID" "$WEB_PID" 2>/dev/null || true
    wait "$API_PID" "$WEB_PID" 2>/dev/null || true
    echo "[runner] Done."
}
trap cleanup EXIT INT TERM

echo ""
echo "============================================"
echo "  Runner stack is ready"
echo "  API:  http://localhost:${API_PORT}"
echo "  Web:  http://localhost:${WEB_PORT}"
echo "============================================"
echo ""
echo "Run eval.sh in another terminal:"
echo "  EVAL_BACKEND_FILTER=vllm bash eval.sh"
echo ""

wait
