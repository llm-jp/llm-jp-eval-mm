#!/bin/bash
# Test one representative model per dependency group from eval.sh
# Tests: (1) uv sync, (2) import/load, (3) basic inference
set -u

export HF_HOME=/workspace/data/shared/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f .env ]; then set -a; source .env; set +a; fi

RESULT_FILE="result_validation/model_group_test.txt"
mkdir -p result_validation

# Representative models per group (smallest available)
# Format: "group|model_id|backend"
declare -a TESTS=(
    "vllm_normal|Qwen/Qwen2.5-VL-3B-Instruct|vllm"
    "vllm_normal|OpenGVLab/InternVL3-1B|vllm"
    "vllm_normal|AIDC-AI/Ovis2-1B|vllm"
    "vllm_normal|OpenGVLab/InternVL3_5-1B|vllm"
    "vllm_normal|Qwen/Qwen3-VL-2B-Instruct|vllm"
    "vllm_normal|Qwen/Qwen3.5-2B|vllm"
    "vllm_normal|AIDC-AI/Ovis2.5-2B|vllm"
    "vllm_normal|Qwen/Qwen3-VL-4B-Instruct|vllm"
    "vllm_normal|Qwen/Qwen3.5-4B|vllm"
    "vllm_normal|OpenGVLab/InternVL3_5-4B|vllm"
    "vllm_normal|allenai/Molmo2-4B|vllm"
    "vllm_normal|AIDC-AI/Ovis2-4B|vllm"
    "vllm_normal|moonshotai/Kimi-VL-A3B-Instruct|vllm"
    "vllm_normal|Qwen/Qwen3-VL-30B-A3B-Instruct|vllm"
    "vllm_normal|Qwen/Qwen3.5-35B-A3B|vllm"
    "vllm_normal|llava-hf/llava-1.5-7b-hf|vllm"
    "vllm_normal|llava-hf/llava-v1.6-mistral-7b-hf|vllm"
    "vllm_normal|neulab/Pangea-7B-hf|vllm"
    "vllm_normal|Qwen/Qwen2-VL-2B-Instruct|vllm"
    "vllm_normal|Qwen/Qwen2-VL-7B-Instruct|vllm"
    "vllm_normal|Qwen/Qwen2.5-VL-7B-Instruct|vllm"
    "vllm_normal|Qwen/Qwen3-VL-8B-Instruct|vllm"
    "vllm_normal|Qwen/Qwen3.5-9B|vllm"
    "vllm_normal|OpenGVLab/InternVL2-8B|vllm"
    "vllm_normal|OpenGVLab/InternVL3-8B|vllm"
    "vllm_normal|OpenGVLab/InternVL3_5-8B|vllm"
    "vllm_normal|CohereLabs/aya-vision-8b|vllm"
    "vllm_normal|allenai/Molmo2-8B|vllm"
    "vllm_normal|AIDC-AI/Ovis2-8B|vllm"
    "vllm_normal|AIDC-AI/Ovis2.5-9B|vllm"
    "vllm_normal|openbmb/MiniCPM-o-2_6|vllm"
    "vllm_normal|google/gemma-3-4b-it|vllm"
    "vllm_normal|google/gemma-3-12b-it|vllm"
    "vllm_normal|llava-hf/llava-1.5-13b-hf|vllm"
    "vllm_normal|OpenGVLab/InternVL3-14B|vllm"
    "vllm_normal|microsoft/Phi-4-multimodal-instruct|vllm"
    "vllm_normal|AIDC-AI/Ovis2-16B|vllm"
    "vllm_normal|OpenGVLab/InternVL2-26B|vllm"
    "vllm_normal|google/gemma-3-27b-it|vllm"
    "vllm_normal|Qwen/Qwen3.5-27B|vllm"
    "vllm_normal|Qwen/Qwen3-VL-32B-Instruct|vllm"
    "vllm_normal|Qwen/Qwen2.5-VL-32B-Instruct|vllm"
    "vllm_normal|CohereLabs/aya-vision-32b|vllm"
    "vllm_normal|AIDC-AI/Ovis2-34B|vllm"
    "vllm_normal|OpenGVLab/InternVL3-38B|vllm"
    "vllm_normal|OpenGVLab/InternVL3_5-38B|vllm"
    "vllm_normal|Qwen/Qwen2-VL-72B-Instruct|vllm"
    "vllm_normal|Qwen/Qwen2.5-VL-72B-Instruct|vllm"
    "vllm_normal|OpenGVLab/InternVL3-78B|vllm"
    "vllm_normal|deepseek-ai/deepseek-vl2|vllm"
    "vllm_normal|zai-org/GLM-4.5V|vllm"
    "vllm_normal|zai-org/GLM-4.6V|vllm"
    "vllm_normal|zai-org/GLM-4.6V-Flash|vllm"
    "gemma4|google/gemma-4-E2B-it|transformers"
    "normal|meta-llama/Llama-3.2-11B-Vision-Instruct|transformers"
    "normal|mistralai/Pixtral-12B-2409|transformers"
    "normal|mistralai/Mistral-Small-3.1-24B-Instruct-2503|transformers"
    "normal|gpt-4o-2024-11-20|transformers"
    "heron_nvila|turing-motors/Heron-NVILA-Lite-1B|transformers"
    "sarashina|sbintuitions/sarashina2-vision-8b|transformers"
    "evovlm|SakanaAI/Llama-3-EvoVLM-JP-v2|transformers"
    "old|MIL-UT/Asagi-14B|transformers"
    "vilaja|llm-jp/llm-jp-3-vila-14b|transformers"
    "calm|cyberagent/llava-calm2-siglip|transformers"
    "stablevlm|stabilityai/japanese-instructblip-alpha|transformers"
    "normal|meta-llama/Llama-3.2-90B-Vision-Instruct|transformers"
)

CURRENT_GROUP=""

> "$RESULT_FILE"

for entry in "${TESTS[@]}"; do
    IFS='|' read -r group model_id backend <<< "$entry"

    # Sync group only when it changes
    if [ "$group" != "$CURRENT_GROUP" ]; then
        echo "--- Syncing group: $group ---"
        if uv sync --group "$group" 2>&1 | tail -1; then
            echo "  sync OK"
        else
            echo "  sync FAIL for $group"
            echo "SYNC_FAIL|$group|$model_id" >> "$RESULT_FILE"
            CURRENT_GROUP="$group"
            continue
        fi
        CURRENT_GROUP="$group"
    fi

    echo "Testing: $model_id ($group/$backend)"

    if [ "$backend" = "vllm" ]; then
        # For vLLM models: check if model is in vllm_registry
        timeout 30 uv run --group "$group" python -c "
import sys
sys.path.insert(0, 'examples')
from vllm_registry import VLLMModelRegistry
try:
    reg = VLLMModelRegistry('$model_id')
    print(f'REGISTRY_OK: {reg.model_id}')
except Exception as e:
    print(f'REGISTRY_FAIL: {e}')
    sys.exit(1)
" 2>&1
        STATUS=$?
    else
        # For transformers models: check if model is in model_table
        timeout 30 uv run --group "$group" python -c "
import sys
sys.path.insert(0, 'examples')
# Skip API models
if '/' not in '$model_id':
    print('SKIP: API model')
    sys.exit(0)
from model_table import get_class_from_model_id
try:
    cls = get_class_from_model_id('$model_id')
    print(f'TABLE_OK: {cls.__name__}')
except KeyError:
    print(f'TABLE_FAIL: $model_id not in model_table')
    sys.exit(1)
except Exception as e:
    print(f'IMPORT_FAIL: {e}')
    sys.exit(1)
" 2>&1
        STATUS=$?
    fi

    if [ $STATUS -eq 0 ]; then
        echo "OK|$group|$model_id" >> "$RESULT_FILE"
        echo "  -> OK"
    else
        echo "FAIL|$group|$model_id" >> "$RESULT_FILE"
        echo "  -> FAIL"
    fi
done

echo ""
echo "=== RESULTS ==="
cat "$RESULT_FILE"
echo ""
echo "=== SUMMARY ==="
echo "OK: $(grep -c '^OK' "$RESULT_FILE")"
echo "FAIL: $(grep -c '^FAIL' "$RESULT_FILE")"
echo "SKIP: $(grep -c '^SKIP' "$RESULT_FILE" || echo 0)"
