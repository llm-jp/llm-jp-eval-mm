#!/usr/bin/env bash
set -euo pipefail

# Run unit tests while controlling HF caches to avoid filling disk.
# 1) metrics first (single shot)
# 2) tasks one-by-one, clearing caches between files

# Prepare HF cache roots
if [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="$(mktemp -d -t hfhome.XXXXXX)"
  CLEAN_HF_HOME=1
else
  CLEAN_HF_HOME=0
fi
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

clear_hf_caches() {
  rm -rf "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" || true
  mkdir -p "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"
}

echo "Clearing HF caches before task tests..."
clear_hf_caches

echo "Running task tests module-by-module with cache clearing..."
# Collect only files that actually contain tests (avoid __init__.py)
TASK_TEST_FILES=()
for f in src/eval_mm/tasks/*.py; do
  if grep -qE "def[[:space:]]+test_" "$f"; then
    TASK_TEST_FILES+=("$f")
  fi
done

if [[ ${#TASK_TEST_FILES[@]} -eq 0 ]]; then
  echo "No task test files found. Skipping."
else
  for f in "${TASK_TEST_FILES[@]}"; do
    echo "--- pytest ${f}"
    uv run --group dev pytest "${f}"
    clear_hf_caches
  done
fi

if [[ "${CLEAN_HF_HOME}" -eq 1 ]]; then
  rm -rf "${HF_HOME}"
fi
