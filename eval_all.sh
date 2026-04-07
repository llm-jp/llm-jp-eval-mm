# Set CUDA devices
set -eux  # エラーが発生したらスクリプトを停止する

#export CUDA_VISIBLE_DEVICES=0

# Model name to group name mapping
# Models that require transformers backend (not supported by vLLM)
# For vLLM-supported models, use eval_with_vllm.sh instead.
declare -A MODEL_GROUP_MAP=(
    # Llama-3.2-Vision (removed from vLLM in v0.10.2)
    ["meta-llama/Llama-3.2-11B-Vision-Instruct"]="normal"
    ["meta-llama/Llama-3.2-90B-Vision-Instruct"]="normal"
    # Pixtral (uses its own vLLM adapter with tokenizer_mode=mistral via pixtral.py)
    ["mistralai/Pixtral-12B-2409"]="vllm_normal"
    # llava-calm2-siglip (custom, needs transformers==4.45.0)
    ["cyberagent/llava-calm2-siglip"]="calm"
    # Sarashina2-Vision (custom, needs transformers==4.47.0)
    ["sbintuitions/sarashina2-vision-8b"]="sarashina"
    ["sbintuitions/sarashina2-vision-14b"]="sarashina"
    # EvoVLM (custom Mantis-based)
    ["SakanaAI/Llama-3-EvoVLM-JP-v2"]="evovlm"
    # llm-jp-3-vila (VILA-based, custom)
    ["llm-jp/llm-jp-3-vila-14b"]="vilaja"
    # Heron-NVILA (custom NVILA)
    ["turing-motors/Heron-NVILA-Lite-1B"]="heron_nvila"
    ["turing-motors/Heron-NVILA-Lite-2B"]="heron_nvila"
    ["turing-motors/Heron-NVILA-Lite-15B"]="heron_nvila"
    ["turing-motors/Heron-NVILA-Lite-33B"]="heron_nvila"
    # Asagi (custom LLaVA, needs transformers<4.50)
    ["MIL-UT/Asagi-14B"]="old"
    # Stability AI (old models)
    ["stabilityai/japanese-instructblip-alpha"]="stablevlm"
    ["stabilityai/japanese-stable-vlm"]="stablevlm"
    # GPT-4o (API-based)
    ["gpt-4o-2024-11-20"]="normal"
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
            uv run --group $model_group  python examples/sample.py \
                --model_id "$model_name" \
                --task_id "$task" \
                --metrics $METRIC \
                --judge_model "gpt-4.1-2025-04-14" \
                --result_dir "$RESULT_DIR"
        done
    done
done

echo "All evaluations are done."
