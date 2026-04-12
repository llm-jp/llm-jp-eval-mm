#!/usr/bin/env bash
# Shared configuration for llm-jp-eval-mm TSUBAME 4.0 scripts.
# Source this file from other scripts: source "$(dirname "$0")/config.sh"
#
# Environment-specific settings (paths, credentials) are loaded from .env.
# This file only contains shared definitions (models, tasks, metrics).

# ============================================================================
# Load .env from project root
# ============================================================================
_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${_PROJECT_ROOT}/.env" ]; then
    set -a; source "${_PROJECT_ROOT}/.env"; set +a
fi

# ============================================================================
# Required environment variables (set these in .env)
# ============================================================================
# PROJECT_DIR     — project clone location (e.g. /gs/bs/.../eval-mm)
# RESULT_DIR      — evaluation results output
# LOG_DIR         — job logs
# HF_HOME         — HuggingFace cache
# TSUBAME_GROUP   — SGE group (-g)
#
# Optional:
# TSUBAME_RESOURCE — SGE resource type (default: node_f)
# TSUBAME_H_RT     — elapsed time limit (default: 24:00:00)
# UV_CACHE_DIR    — uv cache
# VLLM_CACHE_DIR  — vLLM cache
# JUDGE_MODEL     — LLM judge model

PROJECT_DIR="${PROJECT_DIR:-${_PROJECT_ROOT}}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/result}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
TSUBAME_GROUP="${TSUBAME_GROUP:-}"
TSUBAME_RESOURCE="${TSUBAME_RESOURCE:-node_f}"
TSUBAME_H_RT="${TSUBAME_H_RT:-24:00:00}"

# ============================================================================
# GPU / evaluation settings
# ============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE=4
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.1-2025-11-13}"

# ============================================================================
# Tasks & Metrics (same as eval.sh)
# ============================================================================
declare -a TASK_LIST=(
    "japanese-heron-bench"
    "ja-vlm-bench-in-the-wild"
    "ja-vg-vqa-500"
    "jmmmu"
    "ja-multi-image-vqa"
    "jdocqa"
    "mmmu"
    "llava-bench-in-the-wild"
    "cvqa"
    "cc-ocr"
    "mecha-ja"
    "ai2d"
    "docvqa"
    "infographicvqa"
    "textvqa"
    "chartqa"
    "chartqapro"
    # "mathvista"  # 一時的に除外
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
    ["mecha-ja"]="mecha-ja"
    ["cc-ocr"]="cc-ocr"
    ["ai2d"]="ai2d"
    ["cvqa"]="substring-match"
    ["docvqa"]="substring-match"
    ["infographicvqa"]="substring-match"
    ["textvqa"]="substring-match"
    ["chartqa"]="substring-match"
    ["chartqapro"]="substring-match"
    # ["mathvista"]="mathvista"  # 一時的に除外
    ["okvqa"]="substring-match"
)

# Build comma-separated lists for multi-task script
TASK_CSV=$(IFS=,; echo "${TASK_LIST[*]}")
METRIC_CSV=""
for _t in "${TASK_LIST[@]}"; do
    METRIC_CSV="${METRIC_CSV:+$METRIC_CSV,}${METRIC_MAP[$_t]}"
done

# ============================================================================
# Model list — format: "model_id|group|backend|tp"
# Same as eval.sh. Comment out models you don't want to run.
# ============================================================================
declare -a MODEL_LIST=(
    # ~1B (tp=1)
    "OpenGVLab/InternVL3-1B|vllm_normal|vllm|1"
    "OpenGVLab/InternVL3_5-1B|vllm_normal|vllm|1"
    "AIDC-AI/Ovis2-1B|vllm_normal|vllm|1"
    "turing-motors/Heron-NVILA-Lite-1B|heron_nvila|transformers"
    # ~2B (tp=1)
    "OpenGVLab/InternVL3-2B|vllm_normal|vllm|1"
    "OpenGVLab/InternVL3_5-2B|vllm_normal|vllm|1"
    "Qwen/Qwen2-VL-2B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3-VL-2B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3.5-2B|vllm_normal|vllm|1"
    "AIDC-AI/Ovis2-2B|vllm_normal|vllm|1"
    "AIDC-AI/Ovis2.5-2B|vllm_normal|vllm|1"
    "google/gemma-4-E2B-it|gemma4|transformers"
    "turing-motors/Heron-NVILA-Lite-2B|heron_nvila|transformers"
    # ~3-4B (tp=1)
    "Qwen/Qwen2.5-VL-3B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3-VL-4B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3.5-4B|vllm_normal|vllm|1"
    "OpenGVLab/InternVL3_5-4B|vllm_normal|vllm|1"
    "google/gemma-3-4b-it|vllm_normal|vllm|1"
    "google/gemma-4-E4B-it|gemma4|transformers"
    "allenai/Molmo2-4B|vllm_normal|vllm|1"
    "AIDC-AI/Ovis2-4B|vllm_normal|vllm|1"
    "moonshotai/Kimi-VL-A3B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3-VL-30B-A3B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3.5-35B-A3B|vllm_normal|vllm|1"
    "google/gemma-4-26B-A4B-it|gemma4|transformers"
    # ~7-9B (tp=1)
    "llava-hf/llava-1.5-7b-hf|vllm_normal|vllm|1"
    "llava-hf/llava-v1.6-mistral-7b-hf|vllm_normal|vllm|1"
    "neulab/Pangea-7B-hf|vllm_normal|vllm|1"
    "Qwen/Qwen2-VL-7B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen2.5-VL-7B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3-VL-8B-Instruct|vllm_normal|vllm|1"
    "Qwen/Qwen3.5-9B|vllm_normal|vllm|1"
    "OpenGVLab/InternVL2-8B|vllm_normal|vllm|1"
    "OpenGVLab/InternVL3-8B|vllm_normal|vllm|1"
    "OpenGVLab/InternVL3_5-8B|vllm_normal|vllm|1"
    "CohereLabs/aya-vision-8b|vllm_normal|vllm|1"
    "allenai/Molmo2-8B|vllm_normal|vllm|1"
    "AIDC-AI/Ovis2-8B|vllm_normal|vllm|1"
    "AIDC-AI/Ovis2.5-9B|vllm_normal|vllm|1"
    "sbintuitions/sarashina2-vision-8b|sarashina|transformers"
    "SakanaAI/Llama-3-EvoVLM-JP-v2|evovlm|transformers"
    "openbmb/MiniCPM-o-2_6|vllm_normal|vllm|1"
    # ~11-15B (tp=1)
    "meta-llama/Llama-3.2-11B-Vision-Instruct|normal|transformers"
    "mistralai/Pixtral-12B-2409|vllm_normal|transformers"
    "google/gemma-3-12b-it|vllm_normal|vllm|1"
    "llava-hf/llava-1.5-13b-hf|vllm_normal|vllm|1"
    "OpenGVLab/InternVL3-14B|vllm_normal|vllm|1"
    "MIL-UT/Asagi-14B|old|transformers"
    "sbintuitions/sarashina2-vision-14b|sarashina|transformers"
    "microsoft/Phi-4-multimodal-instruct|vllm_normal|vllm|1"
    "turing-motors/Heron-NVILA-Lite-15B|heron_nvila|transformers"
    "AIDC-AI/Ovis2-16B|vllm_normal|vllm|2"
    # ~24-34B (tp=2 or tp=4)
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503|vllm_normal|transformers"
    "OpenGVLab/InternVL2-26B|vllm_normal|vllm|2"
    "google/gemma-3-27b-it|vllm_normal|vllm|2"
    "Qwen/Qwen3.5-27B|vllm_normal|vllm|2"
    "Qwen/Qwen3-VL-32B-Instruct|vllm_normal|vllm|2"
    "Qwen/Qwen2.5-VL-32B-Instruct|vllm_normal|vllm|2"
    "CohereLabs/aya-vision-32b|vllm_normal|vllm|2"
    "google/gemma-4-31B-it|gemma4|transformers"
    # "turing-motors/Heron-NVILA-Lite-33B|heron_nvila|transformers"  # 33B: 1GPU では VRAM 不足
    "AIDC-AI/Ovis2-34B|vllm_normal|vllm|2"
    "OpenGVLab/InternVL3-38B|vllm_normal|vllm|2"
    "OpenGVLab/InternVL3_5-38B|vllm_normal|vllm|2"
    # ~72B+ (tp=4)
    "Qwen/Qwen2-VL-72B-Instruct|vllm_normal|vllm|4"
    "Qwen/Qwen2.5-VL-72B-Instruct|vllm_normal|vllm|4"
    "OpenGVLab/InternVL3-78B|vllm_normal|vllm|4"
    # "meta-llama/Llama-3.2-90B-Vision-Instruct|normal|transformers"  # 90B: 1GPU では VRAM 不足
    "deepseek-ai/deepseek-vl2|vllm_normal|vllm|4"
    # Size unknown / special
    "zai-org/GLM-4.5V|vllm_normal|vllm"
    "zai-org/GLM-4.6V|vllm_normal|vllm"
    "zai-org/GLM-4.6V-Flash|vllm_normal|vllm"
    "cyberagent/llava-calm2-siglip|calm|transformers"
    "stabilityai/japanese-instructblip-alpha|stablevlm|transformers"
    "stabilityai/japanese-stable-vlm|stablevlm|transformers"
    # API-based (skip on TSUBAME — no internet in compute nodes)
    # "gpt-4o-2024-11-20|normal|transformers"
)
