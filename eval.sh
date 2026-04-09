#!/usr/bin/env bash
# Unified evaluation script for all models.
# Models are ordered from smallest to largest.
# Automatically selects the correct backend (vLLM or transformers) per model.
set -eux

# Load environment variables (.env) for HF_TOKEN, API keys, etc.
if [ -f .env ]; then
    set -a; source .env; set +a
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
TENSOR_PARALLEL_SIZE=4
JUDGE_MODEL="gpt-4.1-2025-04-14"
RESULT_DIR="result"

# ============================================================================
# Model list (ordered by parameter count, smallest first)
# Format: "model_id|group|backend"
#   backend = "vllm"         -> sample_vllm.py (vllm_registry.py)
#   backend = "transformers"  -> sample.py     (model_table.py)
# ============================================================================
declare -a MODEL_LIST=(
    # ~1B
    "OpenGVLab/InternVL3-1B|vllm_normal|vllm"
    "OpenGVLab/InternVL3_5-1B|vllm_normal|vllm"
    "AIDC-AI/Ovis2-1B|vllm_normal|vllm"
    "turing-motors/Heron-NVILA-Lite-1B|heron_nvila|transformers"
    # ~2B
    "OpenGVLab/InternVL3-2B|vllm_normal|vllm"
    "OpenGVLab/InternVL3_5-2B|vllm_normal|vllm"
    "Qwen/Qwen2-VL-2B-Instruct|vllm_normal|vllm"
    "Qwen/Qwen3-VL-2B-Instruct|vllm_normal|vllm"
    "Qwen/Qwen3.5-2B|vllm_normal|vllm"
    "AIDC-AI/Ovis2-2B|vllm_normal|vllm"
    "AIDC-AI/Ovis2.5-2B|vllm_normal|vllm"
    "google/gemma-4-E2B-it|gemma4|transformers"
    "turing-motors/Heron-NVILA-Lite-2B|heron_nvila|transformers"
    # ~3-4B
    "Qwen/Qwen2.5-VL-3B-Instruct|vllm_normal|vllm"
    "Qwen/Qwen3-VL-4B-Instruct|vllm_normal|vllm"
    "Qwen/Qwen3.5-4B|vllm_normal|vllm"
    "OpenGVLab/InternVL3_5-4B|vllm_normal|vllm"
    "google/gemma-3-4b-it|vllm_normal|vllm"
    "google/gemma-4-E4B-it|gemma4|transformers"
    "allenai/Molmo2-4B|vllm_normal|vllm"
    "AIDC-AI/Ovis2-4B|vllm_normal|vllm"
    "moonshotai/Kimi-VL-A3B-Instruct|vllm_normal|vllm"
    "Qwen/Qwen3-VL-30B-A3B-Instruct|vllm_normal|vllm"
    "Qwen/Qwen3.5-35B-A3B|vllm_normal|vllm"
    "google/gemma-4-26B-A4B-it|gemma4|transformers"
    # ~7-9B
    "llava-hf/llava-1.5-7b-hf|vllm_normal|vllm"
    "llava-hf/llava-v1.6-mistral-7b-hf|vllm_normal|vllm"
    "neulab/Pangea-7B-hf|vllm_normal|vllm"
    "Qwen/Qwen2-VL-7B-Instruct|vllm_normal|vllm"
    "Qwen/Qwen2.5-VL-7B-Instruct|vllm_normal|vllm"
    "Qwen/Qwen3-VL-8B-Instruct|vllm_normal|vllm"
    "Qwen/Qwen3.5-9B|vllm_normal|vllm"
    "OpenGVLab/InternVL2-8B|vllm_normal|vllm"
    "OpenGVLab/InternVL3-8B|vllm_normal|vllm"
    "OpenGVLab/InternVL3_5-8B|vllm_normal|vllm"
    "CohereLabs/aya-vision-8b|vllm_normal|vllm"
    "allenai/Molmo2-8B|vllm_normal|vllm"
    "AIDC-AI/Ovis2-8B|vllm_normal|vllm"
    "AIDC-AI/Ovis2.5-9B|vllm_normal|vllm"
    "sbintuitions/sarashina2-vision-8b|sarashina|transformers"
    "SakanaAI/Llama-3-EvoVLM-JP-v2|evovlm|transformers"
    "openbmb/MiniCPM-o-2_6|vllm_normal|vllm"
    # ~11-15B
    "meta-llama/Llama-3.2-11B-Vision-Instruct|normal|transformers"
    "mistralai/Pixtral-12B-2409|vllm_normal|transformers"
    "google/gemma-3-12b-it|vllm_normal|vllm"
    "llava-hf/llava-1.5-13b-hf|vllm_normal|vllm"
    "OpenGVLab/InternVL3-14B|vllm_normal|vllm"
    "MIL-UT/Asagi-14B|old|transformers"
    "sbintuitions/sarashina2-vision-14b|sarashina|transformers"
    "llm-jp/llm-jp-3-vila-14b|vilaja|transformers"
    "microsoft/Phi-4-multimodal-instruct|vllm_normal|vllm"
    "turing-motors/Heron-NVILA-Lite-15B|heron_nvila|transformers"
    "AIDC-AI/Ovis2-16B|vllm_normal|vllm"
    # ~24-34B
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503|vllm_normal|transformers"
    "OpenGVLab/InternVL2-26B|vllm_normal|vllm"
    "google/gemma-3-27b-it|vllm_normal|vllm"
    "Qwen/Qwen3.5-27B|vllm_normal|vllm"
    "Qwen/Qwen3-VL-32B-Instruct|vllm_normal|vllm"
    "Qwen/Qwen2.5-VL-32B-Instruct|vllm_normal|vllm"
    "CohereLabs/aya-vision-32b|vllm_normal|vllm"
    "google/gemma-4-31B-it|gemma4|transformers"
    "turing-motors/Heron-NVILA-Lite-33B|heron_nvila|transformers"
    "AIDC-AI/Ovis2-34B|vllm_normal|vllm"
    "OpenGVLab/InternVL3-38B|vllm_normal|vllm"
    "OpenGVLab/InternVL3_5-38B|vllm_normal|vllm"
    # ~72B+
    "Qwen/Qwen2-VL-72B-Instruct|vllm_normal|vllm"
    "Qwen/Qwen2.5-VL-72B-Instruct|vllm_normal|vllm"
    "OpenGVLab/InternVL3-78B|vllm_normal|vllm"
    "meta-llama/Llama-3.2-90B-Vision-Instruct|normal|transformers"
    "deepseek-ai/deepseek-vl2|vllm_normal|vllm"
    # Size unknown / special
    "zai-org/GLM-4.5V|vllm_normal|vllm"
    "zai-org/GLM-4.6V|vllm_normal|vllm"
    "zai-org/GLM-4.6V-Flash|vllm_normal|vllm"
    "cyberagent/llava-calm2-siglip|calm|transformers"
    "stabilityai/japanese-instructblip-alpha|stablevlm|transformers"
    "stabilityai/japanese-stable-vlm|stablevlm|transformers"
    # API-based
    "gpt-4o-2024-11-20|normal|transformers"
)

# ============================================================================
# Tasks & Metrics
# ============================================================================
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

# ============================================================================
# Main evaluation loop (models run in order: smallest first)
# ============================================================================
for task in "${task_list[@]}"; do
    METRIC=${METRIC_MAP[$task]}
    for entry in "${MODEL_LIST[@]}"; do
        IFS='|' read -r model_name model_group backend <<< "$entry"
        uv sync --group "$model_group"

        if [ "$backend" = "vllm" ]; then
            uv run --group "$model_group" python examples/sample_vllm.py \
                --model_id "$model_name" \
                --task_id "$task" \
                --metrics "$METRIC" \
                --judge_model "$JUDGE_MODEL" \
                --result_dir "$RESULT_DIR" \
                --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
                --inference_only
        else
            uv run --group "$model_group" python examples/sample.py \
                --model_id "$model_name" \
                --task_id "$task" \
                --metrics "$METRIC" \
                --judge_model "$JUDGE_MODEL" \
                --result_dir "$RESULT_DIR"
        fi
    done
done

echo "All evaluations are done."
