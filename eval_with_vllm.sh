# Set CUDA devices
set -eux  # エラーが発生したらスクリプトを停止する

#export CUDA_VISIBLE_DEVICES=0

# Model name to group name mapping
declare -A MODEL_GROUP_MAP=(
    ["Qwen/Qwen2.5-VL-3B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-7B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-32B-Instruct"]="normal"
    # ["Qwen/Qwen2.5-VL-72B-Instruct"]="normal"
    ["google/gemma-3-4b-it"]="normal"
    ["google/gemma-3-12b-it"]="normal"
    ["google/gemma-3-27b-it"]="normal"
)

# Task list
declare -a task_list=(
    "japanese-heron-bench"
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
            uv sync --group vllm_normal
            uv run --group vllm_normal  python examples/sample_vllm.py \
                --model_id "$model_name" \
                --task_id "$task" \
                --metrics "$METRIC" \
                --judge_model "gpt-4o-2024-11-20" \
                --result_dir "$RESULT_DIR" \
                --inference_only
        done
    done
done

echo "All evaluations are done."
