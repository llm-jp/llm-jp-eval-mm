#!/usr/bin/env bash
# eval.sh — 統合評価スクリプト: 全モデル × 全タスク
# Usage: bash eval.sh
#
# - Transformers バックエンド (examples/sample.py) と vLLM バックエンド (examples/sample_vllm.py) を順に実行
# - モデルはサイズ順（小→大）に実行し、環境・スクリプトの動作を段階的に検証
# - 既に result/ に結果がある場合はスキップされる（--overwrite なし）
#
# 前提条件:
# - HF_TOKEN が .env に設定済み（gated repo アクセスに必要）
# - GPU: A100 40GB x4 想定、tensor_parallel_size=4
# - uv がインストール済み
#
# スキップ対象:
# - Llama 4 Scout (8+ GPU 必要)
# - Phi-4-reasoning-vision (vLLM 未対応)

set -eux

RESULT_DIR="result"
JUDGE_MODEL="gpt-4.1-2025-04-14"

# ============================================================
# Task list (21 tasks)
# ============================================================
declare -a TASK_LIST=(
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

# ============================================================
# Metrics mapping
# ============================================================
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

# ============================================================
# Part 1: Transformers backend — model → env group mapping
# Sorted by model size (smallest first)
# ============================================================
declare -A TRANSFORMERS_MODEL_GROUP_MAP=(
    # --- ~1B ---
    ["google/gemma-3-1b-it"]="normal"
    ["OpenGVLab/InternVL3-1B"]="normal"
    ["turing-motors/Heron-NVILA-Lite-1B"]="heron_nvila"

    # --- ~2B ---
    ["OpenGVLab/InternVL3-2B"]="normal"
    ["Qwen/Qwen2-VL-2B-Instruct"]="normal"
    ["turing-motors/Heron-NVILA-Lite-2B"]="heron_nvila"

    # --- ~3-4B ---
    ["Qwen/Qwen2.5-VL-3B-Instruct"]="normal"
    ["google/gemma-3-4b-it"]="normal"

    # --- ~7B ---
    ["llava-hf/llava-1.5-7b-hf"]="normal"
    ["llava-hf/llava-v1.6-mistral-7b-hf"]="normal"
    ["SakanaAI/EvoVLM-JP-v1-7B"]="evovlm"
    ["internlm/internlm-xcomposer2d5-7b"]="old"
    ["neulab/Pangea-7B-hf"]="sarashina"
    ["Qwen/Qwen2-VL-7B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-7B-Instruct"]="normal"

    # --- ~8-9B ---
    ["OpenGVLab/InternVL2-8B"]="normal"
    ["OpenGVLab/InternVL3-8B"]="normal"
    ["sbintuitions/sarashina2-vision-8b"]="sarashina"
    ["CohereLabs/aya-vision-8b"]="normal"
    ["SakanaAI/Llama-3-EvoVLM-JP-v2"]="evovlm"
    ["AXCXEPT/Llama-3-EZO-VLM-1"]="evovlm"
    ["OpenGVLab/InternVL3-9B"]="normal"

    # --- ~11-15B ---
    ["meta-llama/Llama-3.2-11B-Vision-Instruct"]="normal"
    ["Kendamarron/Llama-3.2-11B-Vision-Instruct-Swallow-8B-Merge"]="normal"
    ["mistralai/Pixtral-12B-2409"]="pixtral"
    ["google/gemma-3-12b-it"]="normal"
    ["llava-hf/llava-1.5-13b-hf"]="normal"
    ["Efficient-Large-Model/VILA1.5-13b"]="vilaja"
    ["microsoft/Phi-4-multimodal-instruct"]="phi"
    ["OpenGVLab/InternVL3-14B"]="normal"
    ["llm-jp/llm-jp-3-vila-14b"]="vilaja"
    ["MIL-UT/Asagi-14B"]="normal"
    ["sbintuitions/sarashina2-vision-14b"]="sarashina"
    ["stabilityai/japanese-instructblip-alpha"]="normal"
    ["stabilityai/japanese-stable-vlm"]="normal"
    ["cyberagent/llava-calm2-siglip"]="calm"
    ["turing-motors/Heron-NVILA-Lite-15B"]="heron_nvila"

    # --- ~24-34B ---
    ["OpenGVLab/InternVL2-26B"]="normal"
    ["google/gemma-3-27b-it"]="normal"
    ["Qwen/Qwen2.5-VL-32B-Instruct"]="normal"
    ["CohereLabs/aya-vision-32b"]="normal"
    ["turing-motors/Heron-NVILA-Lite-33B"]="heron_nvila"

    # --- ~38B ---
    ["OpenGVLab/InternVL3-38B"]="normal"

    # --- ~72B+ ---
    ["Qwen/Qwen2-VL-72B-Instruct"]="normal"
    ["Qwen/Qwen2.5-VL-72B-Instruct"]="normal"
    ["OpenGVLab/InternVL3-78B"]="normal"
    ["meta-llama/Llama-3.2-90B-Vision-Instruct"]="normal"

    # --- API models ---
    ["gpt-4o-2024-05-13"]="normal"
    ["gpt-4o-2024-11-20"]="normal"
)

# Ordered list for size-ordered execution (bash associative arrays don't preserve order)
declare -a TRANSFORMERS_MODEL_ORDER=(
    # ~1B
    "google/gemma-3-1b-it"
    "OpenGVLab/InternVL3-1B"
    "turing-motors/Heron-NVILA-Lite-1B"
    # ~2B
    "OpenGVLab/InternVL3-2B"
    "Qwen/Qwen2-VL-2B-Instruct"
    "turing-motors/Heron-NVILA-Lite-2B"
    # ~3-4B
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "google/gemma-3-4b-it"
    # ~7B
    "llava-hf/llava-1.5-7b-hf"
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "SakanaAI/EvoVLM-JP-v1-7B"
    "internlm/internlm-xcomposer2d5-7b"
    "neulab/Pangea-7B-hf"
    "Qwen/Qwen2-VL-7B-Instruct"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    # ~8-9B
    "OpenGVLab/InternVL2-8B"
    "OpenGVLab/InternVL3-8B"
    "sbintuitions/sarashina2-vision-8b"
    "CohereLabs/aya-vision-8b"
    "SakanaAI/Llama-3-EvoVLM-JP-v2"
    "AXCXEPT/Llama-3-EZO-VLM-1"
    "OpenGVLab/InternVL3-9B"
    # ~11-15B
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
    "Kendamarron/Llama-3.2-11B-Vision-Instruct-Swallow-8B-Merge"
    "mistralai/Pixtral-12B-2409"
    "google/gemma-3-12b-it"
    "llava-hf/llava-1.5-13b-hf"
    "Efficient-Large-Model/VILA1.5-13b"
    "microsoft/Phi-4-multimodal-instruct"
    "OpenGVLab/InternVL3-14B"
    "llm-jp/llm-jp-3-vila-14b"
    "MIL-UT/Asagi-14B"
    "sbintuitions/sarashina2-vision-14b"
    "stabilityai/japanese-instructblip-alpha"
    "stabilityai/japanese-stable-vlm"
    "cyberagent/llava-calm2-siglip"
    "turing-motors/Heron-NVILA-Lite-15B"
    # ~24-34B
    "OpenGVLab/InternVL2-26B"
    "google/gemma-3-27b-it"
    "Qwen/Qwen2.5-VL-32B-Instruct"
    "CohereLabs/aya-vision-32b"
    "turing-motors/Heron-NVILA-Lite-33B"
    # ~38B
    "OpenGVLab/InternVL3-38B"
    # ~72B+
    "Qwen/Qwen2-VL-72B-Instruct"
    "Qwen/Qwen2.5-VL-72B-Instruct"
    "OpenGVLab/InternVL3-78B"
    "meta-llama/Llama-3.2-90B-Vision-Instruct"
    # API models
    "gpt-4o-2024-05-13"
    "gpt-4o-2024-11-20"
)

# ============================================================
# Part 2: vLLM backend — models from vllm_registry.py
# Sorted by model size (smallest first)
# ============================================================
declare -a VLLM_MODEL_ORDER=(
    # ~1-2B
    "AIDC-AI/Ovis2-1B"
    "AIDC-AI/Ovis2-2B"
    "AIDC-AI/Ovis2.5-2B"
    "openbmb/MiniCPM-o-2_6"
    # ~4B
    "AIDC-AI/Ovis2-4B"
    # ~8-9B
    "AIDC-AI/Ovis2-8B"
    "AIDC-AI/Ovis2.5-9B"
    # ~16B
    "AIDC-AI/Ovis2-16B"
    # ~30B (MoE)
    "Qwen/Qwen3-VL-30B-A3B-Instruct"
    "moonshotai/Kimi-VL-A3B-Instruct"
    # ~34B
    "AIDC-AI/Ovis2-34B"
    # large
    "zai-org/GLM-4.5V"
    "deepseek-ai/deepseek-vl2"
)

# vLLM task list (blink, chartqapro, mathvista excluded for now)
declare -a VLLM_TASK_LIST=(
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
    "docvqa"
    "infographicvqa"
    "textvqa"
    "chartqa"
    "okvqa"
)

# ============================================================
# Run: Transformers backend
# ============================================================
echo "========================================"
echo "Part 1: Transformers backend evaluation"
echo "Models: ${#TRANSFORMERS_MODEL_ORDER[@]}"
echo "Tasks: ${#TASK_LIST[@]}"
echo "========================================"

for model_name in "${TRANSFORMERS_MODEL_ORDER[@]}"; do
    model_group=${TRANSFORMERS_MODEL_GROUP_MAP[$model_name]}
    echo "--- Model: $model_name (group: $model_group) ---"
    uv sync --group "$model_group"
    for task in "${TASK_LIST[@]}"; do
        METRIC=${METRIC_MAP[$task]}
        echo "  Task: $task (metric: $METRIC)"
        uv run --group "$model_group" python examples/sample.py \
            --model_id "$model_name" \
            --task_id "$task" \
            --metrics "$METRIC" \
            --judge_model "$JUDGE_MODEL" \
            --result_dir "$RESULT_DIR"
    done
done

echo "Transformers backend evaluation done."

# ============================================================
# Run: vLLM backend
# ============================================================
echo "========================================"
echo "Part 2: vLLM backend evaluation"
echo "Models: ${#VLLM_MODEL_ORDER[@]}"
echo "Tasks: ${#VLLM_TASK_LIST[@]}"
echo "========================================"

# Activate the vLLM environment
source .uv/vllm_normal-env/bin/activate

for model_name in "${VLLM_MODEL_ORDER[@]}"; do
    echo "--- Model: $model_name (vllm_normal) ---"
    for task in "${VLLM_TASK_LIST[@]}"; do
        METRIC=${METRIC_MAP[$task]}
        echo "  Task: $task (metric: $METRIC)"
        python examples/sample_vllm.py \
            --model_id "$model_name" \
            --task_id "$task" \
            --metrics "$METRIC" \
            --judge_model "$JUDGE_MODEL" \
            --result_dir "$RESULT_DIR" \
            --tensor_parallel_size 4 \
            --inference_only
    done
done

echo "========================================"
echo "All evaluations are done."
echo "========================================"
