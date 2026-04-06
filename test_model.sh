#!/bin/bash
set -eux

# ── Configuration ─────────────────────────────────────────────────
# Override these via environment variables if needed:
#   CUDA_VISIBLE_DEVICES=0,1 bash test_model.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Model name → dependency group mapping
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

# ── Run tests ─────────────────────────────────────────────────────

# Allow filtering via CLI: bash test_model.sh "model_id_substring"
FILTER="${1:-}"

for model_name in "${!MODEL_GROUP_MAP[@]}"; do
    if [[ -n "$FILTER" && "$model_name" != *"$FILTER"* ]]; then
        continue
    fi
    model_group=${MODEL_GROUP_MAP[$model_name]}
    echo "=== Testing $model_name (group: $model_group) ==="
    uv run --group "$model_group" python examples/test_model.py --model_id "$model_name"
done
