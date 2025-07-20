#!/bin/bash
#SBATCH --job-name=llm-jp-eval-mm
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:NUM_GPUS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# module load
. /etc/profile.d/modules.sh
module load cuda/12.1.0
# module load cudnn/9.0.0

# argument
REPO_PATH=$1
model_name=$2
task_name=$3
num_gpus=${4:-1}  # Default to 1 GPU if not specified

# PATH config
export PATH="$HOME/.local/bin:$PATH"
export ROOT_DIR="/home/silviase/"
export DATA_DIR="/data/silviase/"
export HF_HOME="$DATA_DIR/.hf_cache"
export HF_DATASETS_CACHE=$DATA_DIR/datasets
export HF_HUB_CACHE=$DATA_DIR/models
export APPTAINER_CACHEDIR="$DATA_DIR/apptainer_cache"

# Environment Variables
export TORCH_COMPILE_DISABLE=1

# NVLink specific configurations
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_NET_GDR_READ=1
export NCCL_P2P_USE_CUDA_MEMCPY=0

# Set CUDA devices
set -eux

# Model name to group name mapping
declare -A MODEL_GROUP_MAP=(
    # ["stabilityai/japanese-instructblip-alpha"]="normal"
    ["stabilityai/japanese-stable-vlm"]="normal"
    ["cyberagent/llava-calm2-siglip"]="calm"
    ["llava-hf/llava-1.5-7b-hf"]="normal"
    ["llava-hf/llava-v1.6-mistral-7b-hf"]="normal"
    ["neulab/Pangea-7B-hf"]="sarashina"
    ["meta-llama/Llama-3.2-11B-Vision-Instruct"]="normal"
    ["meta-llama/Llama-3.2-90B-Vision-Instruct"]="normal"
    # ["OpenGVLab/InternVL2-8B"]="normal"
    # ["OpenGVLab/InternVL2-26B"]="normal"
    ["Qwen/Qwen2-VL-7B-Instruct"]="normal"
    ["Qwen/Qwen2-VL-72B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-3B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-7B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-32B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-72B-Instruct"]="normal"
    ["gpt-4o-2024-11-20"]="normal"
    # ["mistralai/Pixtral-12B-2409"]="pixtral" # ng
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
)

# Define metrics per task
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

# Result directories
RESULT_DIR="result"

cd $REPO_PATH
METRIC=${METRIC_MAP[$task_name]}
model_group=${MODEL_GROUP_MAP[$model_name]}
source .uv/$model_group-env/bin/activate

# Run the evaluation and capture exit code
python examples/sample.py \
    --model_id "$model_name" \
    --task_id "$task_name" \
    --metrics "$METRIC" \
    --judge_model "gpt-4.1-2025-04-14" \
    --result_dir "$RESULT_DIR" \
    --overwrite

EXIT_CODE=$?

# Ring the bell with exit code
if [ -f scripts/ring.sh ]; then
    bash scripts/ring.sh $EXIT_CODE
fi

echo "All evaluations are done. Exit code: $EXIT_CODE"

# Exit with the same code as sample.py
exit $EXIT_CODE