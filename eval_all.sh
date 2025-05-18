# Set CUDA devices
set -eux  # エラーが発生したらスクリプトを停止する

#export CUDA_VISIBLE_DEVICES=0

# Model name to group name mapping
declare -A MODEL_GROUP_MAP=(
    ["stabilityai/japanese-instructblip-alpha"]="normal"
    ["stabilityai/japanese-stable-vlm"]="normal"
    ["cyberagent/llava-calm2-siglip"]="calm"
    ["llava-hf/llava-1.5-7b-hf"]="normal"
    ["llava-hf/llava-v1.6-mistral-7b-hf"]="normal"
    ["neulab/Pangea-7B-hf"]="sarashina"
    ["meta-llama/Llama-3.2-11B-Vision-Instruct"]="normal"
    ["meta-llama/Llama-3.2-90B-Vision-Instruct"]="normal"
    ["OpenGVLab/InternVL2-8B"]="normal"
    ["OpenGVLab/InternVL2-26B"]="normal"
    ["Qwen/Qwen2-VL-7B-Instruct"]="normal"
    ["Qwen/Qwen2-VL-72B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-7B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-72B-Instruct"]="normal"
    ["gpt-4o-2024-11-20"]="normal"
    ["mistralai/Pixtral-12B-2409"]="pixtral"
    ["llm-jp/llm-jp-3-vila-14b"]="vilaja"
    ["Efficient-Large-Model/VILA1.5-13b"]="vilaja"
    ["SakanaAI/Llama-3-EvoVLM-JP-v2"]="evovlm"
    ["google/gemma-3-4b-it"]="normal"
    ["google/gemma-3-12b-it"]="normal"
    ["google/gemma-3-27b-it"]="normal"
    ["sbintuitions/sarashina2-vision-8b"]="sarashina"
    ["sbintuitions/sarashina2-vision-14b"]="sarashina"
    ["microsoft/Phi-4-multimodal-instruct"]="phi"
    ["turing-motors/Heron-NVILA-Lite-15B"]="heron_nvila"
)

# Task list
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
    "cc-ocr"
    "cvqa"
)

# Define metrics per task
declare -A METRIC_MAP=(
    ["japanese-heron-bench"]="heron-bench"
    ["ja-vlm-bench-in-the-wild"]="llm-as-a-judge,rougel"
    ["ja-vg-vqa-500"]="llm-as-a-judge,rougel"
    ["jmmmu"]="jmmmu"
    ["ja-multi-image-vqa"]="llm-as-a-judge,rougel"
    ["jdocqa"]="jdocqa,llm-as-a-judge"
    ["mmmu"]="mmmu"
    ["llava-bench-in-the-wild"]="llm-as-a-judge,rougel"
    ["jic-vqa"]="jic-vqa"
    ["mecha-ja"]="mecha-ja"
    ["cc-ocr"]="cc-ocr"
    ["cvqa"]="substring-match"
)

# Result directories
declare -a result_dir_list=(
    "result"
)

# Main evaluation loop
for RESULT_DIR in "${result_dir_list[@]}"; do
    for task in "${task_list[@]}"; do
        METRIC=${METRIC_MAP[$task]}
        for model_name in "${!MODEL_GROUP_MAP[@]}"; do
            model_group=${MODEL_GROUP_MAP[$model_name]}
            uv sync --group $model_group
            uv run --group $model_group  python examples/sample.py \
                --model_id "$model_name" \
                --task_id "$task" \
                --metrics "$METRIC" \
                --judge_model "gpt-4o-2024-11-20" \
                --result_dir "$RESULT_DIR"
        done
    done
done

echo "All evaluations are done."
