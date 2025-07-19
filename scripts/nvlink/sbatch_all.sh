#!/usr/bin/env bash
# How-to-use: bash scripts/nvlink/sbatch_all.sh
# Submit all models for evaluation
set -euo pipefail

# === Common Settings ===
SCRIPT_PATH="$(dirname "$0")"

## All available models with their GPU requirements
declare -A model_gpu_map=(
    # Small models (1 GPU)
    # ["stabilityai/japanese-instructblip-alpha"]=1
    ["stabilityai/japanese-stable-vlm"]=1
    ["cyberagent/llava-calm2-siglip"]=1
    ["llava-hf/llava-1.5-7b-hf"]=1
    ["llava-hf/llava-v1.6-mistral-7b-hf"]=1
    ["neulab/Pangea-7B-hf"]=1
    ["Qwen/Qwen2-VL-7B-Instruct"]=1
    ["Qwen/Qwen2.5-VL-3B-Instruct"]=1
    ["Qwen/Qwen2.5-VL-7B-Instruct"]=1
    ["SakanaAI/Llama-3-EvoVLM-JP-v2"]=1
    ["google/gemma-3-4b-it"]=1
    ["microsoft/Phi-4-multimodal-instruct"]=1
    ["turing-motors/Heron-NVILA-Lite-1B"]=1
    ["turing-motors/Heron-NVILA-Lite-2B"]=1
    ["MIL-UT/Asagi-14B"]=1
    ["meta-llama/Llama-3.2-11B-Vision-Instruct"]=1
    ["llm-jp/llm-jp-3-vila-14b"]=1
    ["google/gemma-3-12b-it"]=1
    ["sbintuitions/sarashina2-vision-8b"]=1
    ["sbintuitions/sarashina2-vision-14b"]=1
    
    # Medium models (2 GPUs)
    ["turing-motors/Heron-NVILA-Lite-15B"]=2
    ["google/gemma-3-27b-it"]=2
    ["turing-motors/Heron-NVILA-Lite-33B"]=2
    ["Qwen/Qwen2.5-VL-32B-Instruct"]=2

    # Large models (4 GPUs)
    ["meta-llama/Llama-3.2-90B-Vision-Instruct"]=4
    ["Qwen/Qwen2-VL-72B-Instruct"]=4
    ["Qwen/Qwen2.5-VL-72B-Instruct"]=4
)

# === Submit all models ===
echo "🚀 Starting batch submission for all models"
echo "📊 Total models to evaluate: ${#model_gpu_map[@]}"
echo ""

counter=0
for model in "${!model_gpu_map[@]}"; do
    num_gpus=${model_gpu_map[$model]}
    counter=$((counter + 1))
    echo "[$counter/${#model_gpu_map[@]}] Submitting $model with $num_gpus GPU(s)..."
    bash "$SCRIPT_PATH/sbatch.sh" "$model" "$num_gpus"
    echo ""
    sleep 1  # Small delay between submissions
done

echo "✅ All models have been submitted!"
echo "📊 Total models submitted: $counter"