# Set CUDA devices
set -eux  # エラーが発生したらスクリプトを停止する

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Model name to group name mapping
# All models here use the vllm_normal group with the VLLM wrapper (base_vllm.py + vllm_registry.py)
declare -A MODEL_GROUP_MAP=(
    # LLaVA 1.5
    ["llava-hf/llava-1.5-7b-hf"]="vllm_normal"
    ["llava-hf/llava-1.5-13b-hf"]="vllm_normal"
    # LLaVA-NeXT
    ["llava-hf/llava-v1.6-mistral-7b-hf"]="vllm_normal"
    # Pangea (LLaVA-NeXT based)
    ["neulab/Pangea-7B-hf"]="vllm_normal"
    # InternVL2
    ["OpenGVLab/InternVL2-8B"]="vllm_normal"
    ["OpenGVLab/InternVL2-26B"]="vllm_normal"
    # InternVL3
    ["OpenGVLab/InternVL3-1B"]="vllm_normal"
    ["OpenGVLab/InternVL3-2B"]="vllm_normal"
    ["OpenGVLab/InternVL3-8B"]="vllm_normal"
    ["OpenGVLab/InternVL3-14B"]="vllm_normal"
    ["OpenGVLab/InternVL3-38B"]="vllm_normal"
    ["OpenGVLab/InternVL3-78B"]="vllm_normal"
    # Qwen2-VL
    ["Qwen/Qwen2-VL-7B-Instruct"]="vllm_normal"
    ["Qwen/Qwen2-VL-72B-Instruct"]="vllm_normal"
    # Qwen2.5-VL
    ["Qwen/Qwen2.5-VL-3B-Instruct"]="vllm_normal"
    ["Qwen/Qwen2.5-VL-7B-Instruct"]="vllm_normal"
    ["Qwen/Qwen2.5-VL-32B-Instruct"]="vllm_normal"
    ["Qwen/Qwen2.5-VL-72B-Instruct"]="vllm_normal"
    # Gemma-3
    ["google/gemma-3-4b-it"]="vllm_normal"
    ["google/gemma-3-12b-it"]="vllm_normal"
    ["google/gemma-3-27b-it"]="vllm_normal"
    # Phi-4
    ["microsoft/Phi-4-multimodal-instruct"]="vllm_normal"
    # Aya Vision
    ["CohereLabs/aya-vision-8b"]="vllm_normal"
    ["CohereLabs/aya-vision-32b"]="vllm_normal"
    # GLM-4.5V
    ["zai-org/GLM-4.5V"]="vllm_normal"
    # ── Successor models ──
    # Qwen3-VL (Dense)
    ["Qwen/Qwen3-VL-2B-Instruct"]="vllm_normal"
    ["Qwen/Qwen3-VL-4B-Instruct"]="vllm_normal"
    ["Qwen/Qwen3-VL-8B-Instruct"]="vllm_normal"
    ["Qwen/Qwen3-VL-30B-A3B-Instruct"]="vllm_normal"
    ["Qwen/Qwen3-VL-32B-Instruct"]="vllm_normal"
    # Qwen3.5 (natively multimodal)
    ["Qwen/Qwen3.5-2B"]="vllm_normal"
    ["Qwen/Qwen3.5-4B"]="vllm_normal"
    ["Qwen/Qwen3.5-9B"]="vllm_normal"
    ["Qwen/Qwen3.5-27B"]="vllm_normal"
    ["Qwen/Qwen3.5-35B-A3B"]="vllm_normal"
    # InternVL3.5
    ["OpenGVLab/InternVL3_5-1B"]="vllm_normal"
    ["OpenGVLab/InternVL3_5-2B"]="vllm_normal"
    ["OpenGVLab/InternVL3_5-4B"]="vllm_normal"
    ["OpenGVLab/InternVL3_5-8B"]="vllm_normal"
    ["OpenGVLab/InternVL3_5-38B"]="vllm_normal"
    # Molmo2
    ["allenai/Molmo2-4B"]="vllm_normal"
    ["allenai/Molmo2-8B"]="vllm_normal"
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
    "blink"
    "docvqa"
    "infographicvqa"
    "textvqa"
    "chartqa"
    "chartqapro"
    "mathvista"
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
            uv sync --group $model_group
            uv run --group $model_group python examples/sample_vllm.py \
                --model_id "$model_name" \
                --task_id "$task" \
                --metrics "$METRIC" \
                --judge_model "gpt-4.1-2025-04-14" \
                --result_dir "$RESULT_DIR" \
                --tensor_parallel_size 4 \
                --inference_only
        done
    done
done

echo "All evaluations are done."
