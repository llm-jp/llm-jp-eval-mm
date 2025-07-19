#!/bin/bash
# Base script for model testing that can be run both interactively and via sbatch

# argument
REPO_PATH=$1

# PATH config
export PATH="$HOME/.local/bin:$PATH"
export ROOT_DIR="/home/silviase/"
export HF_HOME="$ROOT_DIR/.hf_cache"
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/models
export APPTAINER_CACHEDIR="$ROOT_DIR/apptainer_cache"

# NVLink specific configurations
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_NET_GDR_READ=1
export NCCL_P2P_USE_CUDA_MEMCPY=0

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0
set -eux

# Model name to group name mapping
declare -A MODEL_GROUP_MAP=(
    ["stabilityai/japanese-instructblip-alpha"]="old" # ng transformers<4.50?
    ["stabilityai/japanese-stable-vlm"]="calm" # ok
    ["cyberagent/llava-calm2-siglip"]="calm" # ok 
    ["llava-hf/llava-1.5-7b-hf"]="normal" # ok
    ["llava-hf/llava-v1.6-mistral-7b-hf"]="normal" # ok
    ["neulab/Pangea-7B-hf"]="sarashina" # ok
    ["meta-llama/Llama-3.2-11B-Vision-Instruct"]="normal" # ok 
    # ["OpenGVLab/InternVL2-8B"]="old" # ng transformers<4.50?
    ["Qwen/Qwen2-VL-7B-Instruct"]="normal" # ok 
    ["Qwen/Qwen2.5-VL-7B-Instruct"]="normal" # ok
    # ["gpt-4o-2024-05-13"]="normal"
    ["mistralai/Pixtral-12B-2409"]="pixtral" # ok
    ["llm-jp/llm-jp-3-vila-14b"]="vilaja" # ok
    # ["Efficient-Large-Model/VILA1.5-13b"]="vilaja" # ng
    ["SakanaAI/Llama-3-EvoVLM-JP-v2"]="evovlm" # ok 
    ["google/gemma-3-4b-it"]="normal" # ok 
    ["sbintuitions/sarashina2-vision-8b"]="sarashina" # ok
    ["microsoft/Phi-4-multimodal-instruct"]="phi" # ok 
    ["MIL-UT/Asagi-14B"]="normal" # ok 
    ["turing-motors/Heron-NVILA-Lite-2B"]="old" # ok
)

cd $REPO_PATH

# Test all models
for model_name in "${!MODEL_GROUP_MAP[@]}"; do
    echo "Testing model: $model_name"
    model_group=${MODEL_GROUP_MAP[$model_name]}
    
    # Activate the appropriate virtual environment
    source .uv/$model_group-env/bin/activate
    python examples/test_model.py --model_id "$model_name"

    echo "Completed testing: $model_name"
    echo "---"
done

echo "All model tests completed."