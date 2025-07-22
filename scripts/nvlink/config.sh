#!/bin/bash
# Common configuration file for nvlink scripts

# === Path Configuration ===
export ROOT_DIR="/home/silviase/"
export DATA_DIR="/data/silviase/"
export REPO_PATH="$ROOT_DIR/llm-jp-eval-mm"
export SCRIPT_PATH="$REPO_PATH/scripts/nvlink"

# HuggingFace cache directories
export HF_HOME="$DATA_DIR/.hf_cache"
export HF_DATASETS_CACHE="$DATA_DIR/datasets"
export HF_HUB_CACHE="$DATA_DIR/models"
export APPTAINER_CACHEDIR="$DATA_DIR/apptainer_cache"

# CUDA configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Model List ===
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
    "OpenGVLab/InternVL3-1B"
    "OpenGVLab/InternVL3-2B"
    "OpenGVLab/InternVL3-8B"
    "OpenGVLab/InternVL3-9B"
    "OpenGVLab/InternVL3-14B"
    "OpenGVLab/InternVL3-38B"
    "OpenGVLab/InternVL3-78B"
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
    "CohereLabs/aya-vision-8b"
    "CohereLabs/aya-vision-32b"
)

# === Model to Environment Group Mapping ===
declare -A MODEL_GROUP_MAP=(
    ["stabilityai/japanese-instructblip-alpha"]="old" # ng transformers<4.50?
    ["stabilityai/japanese-stable-vlm"]="calm"
    ["cyberagent/llava-calm2-siglip"]="calm"
    ["llava-hf/llava-1.5-7b-hf"]="normal"
    ["llava-hf/llava-v1.6-mistral-7b-hf"]="normal"
    ["neulab/Pangea-7B-hf"]="sarashina"
    ["meta-llama/Llama-3.2-11B-Vision-Instruct"]="normal"
    ["meta-llama/Llama-3.2-90B-Vision-Instruct"]="normal"
    # ["OpenGVLab/InternVL2-8B"]="normal"
    # ["OpenGVLab/InternVL2-26B"]="normal"
    ["OpenGVLab/InternVL3-1B"]="normal"
    ["OpenGVLab/InternVL3-2B"]="normal"
    ["OpenGVLab/InternVL3-8B"]="normal"
    ["OpenGVLab/InternVL3-9B"]="normal"
    ["OpenGVLab/InternVL3-14B"]="normal"
    ["OpenGVLab/InternVL3-38B"]="normal"
    ["OpenGVLab/InternVL3-78B"]="normal"
    ["Qwen/Qwen2-VL-7B-Instruct"]="normal"
    ["Qwen/Qwen2-VL-72B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-3B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-7B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-32B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-72B-Instruct"]="normal"
    ["gpt-4o-2024-11-20"]="normal"
    ["mistralai/Pixtral-12B-2409"]="pixtral"
    ["llm-jp/llm-jp-3-vila-14b"]="vilaja"
    # ["Efficient-Large-Model/VILA1.5-13b"]="vilaja" # ng
    ["SakanaAI/Llama-3-EvoVLM-JP-v2"]="evovlm"
    ["google/gemma-3-4b-it"]="normal"
    ["google/gemma-3-12b-it"]="normal"
    ["google/gemma-3-27b-it"]="normal"
    ["google/gemma-3-4b-pt"]="normal"
    ["google/gemma-3-12b-pt"]="normal"
    ["google/gemma-3-27b-pt"]="normal"
    # ["tokyotech-llm/gemma3_4b_exp8-checkpoint-50000"]="normal"
    ["sbintuitions/sarashina2-vision-8b"]="sarashina"
    ["sbintuitions/sarashina2-vision-14b"]="sarashina"
    ["microsoft/Phi-4-multimodal-instruct"]="phi"
    ["MIL-UT/Asagi-14B"]="normal"
    ["turing-motors/Heron-NVILA-Lite-1B"]="old"
    ["turing-motors/Heron-NVILA-Lite-2B"]="old"
    ["turing-motors/Heron-NVILA-Lite-15B"]="old"
    ["turing-motors/Heron-NVILA-Lite-33B"]="old"
    ["CohereLabs/aya-vision-8b"]="normal"
    ["CohereLabs/aya-vision-32b"]="normal"
)

# === Model GPU Requirements ===
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
    ["OpenGVLab/InternVL3-1B"]=1
    ["OpenGVLab/InternVL3-2B"]=1
    ["OpenGVLab/InternVL3-8B"]=1
    ["OpenGVLab/InternVL3-9B"]=1
    ["OpenGVLab/InternVL3-14B"]=1
    ["sbintuitions/sarashina2-vision-8b"]=2
    ["sbintuitions/sarashina2-vision-14b"]=2
    ["CohereLabs/aya-vision-8b"]=1

    # Medium models (2 GPUs)
    ["google/gemma-3-27b-it"]=2
    ["turing-motors/Heron-NVILA-Lite-15B"]=2
    ["turing-motors/Heron-NVILA-Lite-33B"]=2
    ["Qwen/Qwen2.5-VL-32B-Instruct"]=2
    ["CohereLabs/aya-vision-32b"]=2
    ["OpenGVLab/InternVL3-38B"]=4
    
    # Large models (4 GPUs)
    ["meta-llama/Llama-3.2-90B-Vision-Instruct"]=4
    ["Qwen/Qwen2-VL-72B-Instruct"]=4
    ["Qwen/Qwen2.5-VL-72B-Instruct"]=4
    ["OpenGVLab/InternVL3-78B"]=8
)

# === Task List ===
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

# === Metrics Mapping ===
declare -A METRIC_MAP=(
    ["japanese-heron-bench"]="heron-bench"
    ["ja-vlm-bench-in-the-wild"]="llm-as-a-judge"
    ["ja-vg-vqa-500"]="llm-as-a-judge"
    ["jmmmu"]="jmmmu"
    ["ja-multi-image-vqa"]="llm-as-a-judge"
    ["jdocqa"]="llm-as-a-judge"
    ["mmmu"]="mmmu"
    ["llava-bench-in-the-wild"]="llm-as-a-judge"
    ["jic-vqa"]="jic-vqa"
    ["mecha-ja"]="mecha-ja"
)

# === Function to load .env file ===
load_env_file() {
    if [ -f "$REPO_PATH/.env" ]; then
        export $(grep -v '^#' "$REPO_PATH/.env" | xargs)
    fi
}