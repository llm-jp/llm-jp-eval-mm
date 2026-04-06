#!/bin/bash
set -euo pipefail

# Set CUDA devices
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Portable cache directories for smoke tests
export DATA_DIR="${DATA_DIR:-artifact/model_smoke_tmp_cache}"
mkdir -p "$DATA_DIR"
export HF_HOME="${HF_HOME:-$DATA_DIR/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DATA_DIR/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$DATA_DIR/models}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$DATA_DIR/apptainer_cache}"

# CUDA configuration
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SMOKE_TEST_MODE="${SMOKE_TEST_MODE:-offline}"

# Model name to group name mapping
declare -A MODEL_GROUP_MAP=(
    ["stabilityai/japanese-instructblip-alpha"]="normal"
    ["stabilityai/japanese-stable-vlm"]="calm"
    ["cyberagent/llava-calm2-siglip"]="calm"
    ["llava-hf/llava-1.5-7b-hf"]="normal"
    ["llava-hf/llava-v1.6-mistral-7b-hf"]="normal"
    ["neulab/Pangea-7B-hf"]="sarashina"
    ["meta-llama/Llama-3.2-11B-Vision-Instruct"]="normal"
    ["OpenGVLab/InternVL2-8B"]="normal"
    ["Qwen/Qwen2-VL-7B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-7B-Instruct"]="normal"
    ["gpt-4o-2024-05-13"]="normal"
    ["mistralai/Pixtral-12B-2409"]="pixtral"
    ["llm-jp/llm-jp-3-vila-14b"]="vilaja"
    ["Efficient-Large-Model/VILA1.5-13b"]="vilaja"
    ["SakanaAI/Llama-3-EvoVLM-JP-v2"]="evovlm"
    ["google/gemma-3-4b-it"]="normal"
    ["sbintuitions/sarashina2-vision-8b"]="sarashina"
    ["microsoft/Phi-4-multimodal-instruct"]="phi"
    ["MIL-UT/Asagi-14B"]="normal"
)

models=("$@")
if [ ${#models[@]} -eq 0 ]; then
    mapfile -t models < <(printf '%s\n' "${!MODEL_GROUP_MAP[@]}" | sort)
fi

for model_name in "${models[@]}"; do
    if [ -z "${MODEL_GROUP_MAP[$model_name]+x}" ]; then
        echo "Unknown model_id: $model_name" >&2
        exit 1
    fi
    model_group=${MODEL_GROUP_MAP[$model_name]}
    uv run --group "$model_group" python examples/test_model.py \
        --model_id "$model_name" \
        --smoke-test-mode "$SMOKE_TEST_MODE"
done
