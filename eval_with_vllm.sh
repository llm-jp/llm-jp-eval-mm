# Set CUDA devices
set -eux  # エラーが発生したらスクリプトを停止する

export CUDA_VISIBLE_DEVICES=0,1

# Model name to group name mapping
declare -A MODEL_GROUP_MAP=(
    ["Qwen/Qwen3-VL-30B-A3B-Instruct"]="vllm_normal"
    # ["moonshotai/Kimi-VL-A3B-Instruct"]="vllm_normal" # 今は動かない
    ["deepseek-ai/deepseek-vl2"]="vllm_normal"
    ["OpenGVLab/InternVL3-1B"]="vllm_normal"
    ["OpenGVLab/InternVL3-2B"]="vllm_normal"
    ["OpenGVLab/InternVL3-8B"]="vllm_normal"
    ["OpenGVLab/InternVL3-14B"]="vllm_normal"
    ["OpenGVLab/InternVL3-38B"]="vllm_normal"
    ["OpenGVLab/InternVL3-78B"]="vllm_normal"
)

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
    "cvqa"
    "cc-ocr"
    "mecha-ja"
    "ai2d"
    # "blink"
    "docvqa"
    "infographicvqa"
    "textvqa"
    "chartqa"
    # "chartqapro"
    # "mathvista"
    "okvqa"
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
    ["cc-ocr"]="cc-ocr"
    ["ai2d"]="ai2d"
    ["blink"]="blink"
    ["cvqa"]="substring-match"
    ["docvqa"]="substring-match"
    ["infographicvqa"]="substring-match"
    ["textvqa"]="substring-match"
    ["chartqa"]="substring-match"
    ["chartqapro"]="substring-match"
    ["mathvista"]="mathvista"
    ["okvqa"]="substring-match"
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
            source .uv/vllm_normal-env/bin/activate
            uv pip list
            python examples/sample_vllm.py \
                --model_id "$model_name" \
                --task_id "$task" \
                --metrics "$METRIC" \
                --judge_model "gpt-4.1-2025-04-14" \
                --result_dir "$RESULT_DIR" \
                --tensor_parallel_size 2 \
                # --inference_only
        done
    done
done

echo "All evaluations are done."
