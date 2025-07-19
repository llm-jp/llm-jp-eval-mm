#!/usr/bin/env bash
# How-to-use: bash scripts/nvlink/sbatch.sh MODEL_NAME [NUM_GPUS]
# Example: bash scripts/nvlink/sbatch.sh "llava-hf/llava-1.5-7b-hf" 1
set -euo pipefail

# === Argument Parsing ===
if [ $# -lt 1 ]; then
    echo "Usage: $0 MODEL_NAME [NUM_GPUS]"
    echo "  MODEL_NAME: The model to evaluate (required)"
    echo "  NUM_GPUS: Number of GPUs to use (optional, auto-detected based on model size)"
    exit 1
fi

MODEL_NAME=$1
NUM_GPUS=${2:-"auto"}

# === Common Settings ===
## Paths
ROOT_DIR="/home/silviase/"
REPO_PATH="$ROOT_DIR/llm-jp-eval-mm"
SCRIPT_PATH="$REPO_PATH/scripts/nvlink"

## All available models
declare -a model_list=(
    "stabilityai/japanese-instructblip-alpha"
    "stabilityai/japanese-stable-vlm"
    "cyberagent/llava-calm2-siglip"
    "llava-hf/llava-1.5-7b-hf"
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "neulab/Pangea-7B-hf"
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
    "meta-llama/Llama-3.2-90B-Vision-Instruct"
    # "OpenGVLab/InternVL2-8B" # ng
    # "OpenGVLab/InternVL2-26B" # ng
    "Qwen/Qwen2-VL-7B-Instruct"
    "Qwen/Qwen2-VL-72B-Instruct"
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen/Qwen2.5-VL-32B-Instruct"
    "Qwen/Qwen2.5-VL-72B-Instruct"
    # "gpt-4o-2024-11-20"
    # "mistralai/Pixtral-12B-2409" # ng
    "llm-jp/llm-jp-3-vila-14b"
    # "Efficient-Large-Model/VILA1.5-13b" # ng
    "SakanaAI/Llama-3-EvoVLM-JP-v2"
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
    # "google/gemma-3-4b-pt"
    # "google/gemma-3-12b-pt"
    # "google/gemma-3-27b-pt"
    # "tokyotech-llm/gemma3_4b_exp8-checkpoint-5000"
    # "tokyotech-llm/gemma3_4b_exp8-checkpoint-50000"
    "sbintuitions/sarashina2-vision-8b"
    "sbintuitions/sarashina2-vision-14b"
    "microsoft/Phi-4-multimodal-instruct"
    "turing-motors/Heron-NVILA-Lite-1B"
    "turing-motors/Heron-NVILA-Lite-2B"
    "turing-motors/Heron-NVILA-Lite-15B"
    "turing-motors/Heron-NVILA-Lite-33B"
    "MIL-UT/Asagi-14B"
)

## Model GPU requirements
declare -A model_gpu_map=(
    # Small models (1 GPU)
    ["stabilityai/japanese-instructblip-alpha"]=1
    ["stabilityai/japanese-stable-vlm"]=1
    ["cyberagent/llava-calm2-siglip"]=1
    ["llava-hf/llava-1.5-7b-hf"]=1
    ["llava-hf/llava-v1.6-mistral-7b-hf"]=1
    ["neulab/Pangea-7B-hf"]=1
    ["Qwen/Qwen2-VL-7B-Instruct"]=1
    ["Qwen/Qwen2.5-VL-3B-Instruct"]=1
    ["Qwen/Qwen2.5-VL-7B-Instruct"]=1
    ["SakanaAI/Llama-3-EvoVLM-JP-v2"]=1
    ["microsoft/Phi-4-multimodal-instruct"]=1
    ["turing-motors/Heron-NVILA-Lite-1B"]=1
    ["turing-motors/Heron-NVILA-Lite-2B"]=1
    ["MIL-UT/Asagi-14B"]=1
    ["meta-llama/Llama-3.2-11B-Vision-Instruct"]=1
    ["llm-jp/llm-jp-3-vila-14b"]=1
    ["google/gemma-3-4b-it"]=1
    ["google/gemma-3-12b-it"]=1
    ["sbintuitions/sarashina2-vision-8b"]=1
    ["sbintuitions/sarashina2-vision-14b"]=1

    # Medium models (2 GPUs)
    ["google/gemma-3-27b-it"]=2
    ["turing-motors/Heron-NVILA-Lite-15B"]=2
    ["turing-motors/Heron-NVILA-Lite-33B"]=2
    ["Qwen/Qwen2.5-VL-32B-Instruct"]=2
    
    # Large models (4 GPUs)
    ["meta-llama/Llama-3.2-90B-Vision-Instruct"]=4
    ["Qwen/Qwen2-VL-72B-Instruct"]=4
    ["Qwen/Qwen2.5-VL-72B-Instruct"]=4
)

## Task list
declare -a task_list=(
    "japanese-heron-bench"
    "ja-vlm-bench-in-the-wild"
    "ja-vg-vqa-500"
    "jmmmu"
    "ja-multi-image-vqa"
    "jdocqa"
    "mmmu"
    "llava-bench-in-the-wild"
    "jic-vqa"
    "mecha-ja"
)

# === Validate model name ===
model_found=false
for model in "${model_list[@]}"; do
    if [ "$MODEL_NAME" = "$model" ]; then
        model_found=true
        break
    fi
done

# === Check if model is valid ===
if [ "$model_found" = false ]; then
    echo "❌ Model '$MODEL_NAME' is not in the available model list."
    echo "📋 Available models:"
    for model in "${model_list[@]}"; do
        echo "   - $model"
    done
    exit 1
fi

# === Auto-detect GPU count if needed ===
if [ "$NUM_GPUS" = "auto" ]; then
    NUM_GPUS=${model_gpu_map[$MODEL_NAME]:-1}  # Default to 1 if not in map
    echo "🔍 Auto-detected GPU count: $NUM_GPUS"
fi

# Validate GPU count
if ! [[ "$NUM_GPUS" =~ ^[1-4]$ ]]; then
    echo "❌ NUM_GPUS must be between 1 and 4"
    exit 1
fi

# === Submit jobs ===
echo "🚀 Submitting jobs for model: $MODEL_NAME"
echo "🖥️  Number of GPUs: $NUM_GPUS"
echo "📋 Number of tasks: ${#task_list[@]}"
echo ""

for task in "${task_list[@]}"; do
    mkdir -p "$REPO_PATH/outputs/$MODEL_NAME/llm-jp-eval-mm/"
    echo "  📝 Submitting task: $task"
    
    # Create a temporary script with the correct GPU count
    TEMP_SCRIPT=$(mktemp)
    sed "s/NUM_GPUS/$NUM_GPUS/g" "$SCRIPT_PATH/eval.sh" > "$TEMP_SCRIPT"
    chmod +x "$TEMP_SCRIPT"
    
    sbatch \
        --output="$REPO_PATH/outputs/$MODEL_NAME/llm-jp-eval-mm/$task.out" \
        --error="$REPO_PATH/outputs/$MODEL_NAME/llm-jp-eval-mm/$task.err" \
        "$TEMP_SCRIPT" \
        "$REPO_PATH" \
        "$MODEL_NAME" \
        "$task" \
        "$NUM_GPUS"
    
    rm "$TEMP_SCRIPT"
done

echo ""
echo "✅ All tasks submitted successfully for $MODEL_NAME!"
echo "📊 Total jobs submitted: ${#task_list[@]}"