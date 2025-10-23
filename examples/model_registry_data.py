"""Shared model identifier metadata for examples runtime wiring."""

INTERNVL_VLLM_MODELS: tuple[str, ...] = (
    "OpenGVLab/InternVL3-1B",
    "OpenGVLab/InternVL3-2B",
    "OpenGVLab/InternVL3-8B",
    "OpenGVLab/InternVL3-14B",
    "OpenGVLab/InternVL3-38B",
    "OpenGVLab/InternVL3-78B",
)

# OVIS2_VLLM_MODELS: tuple[str, ...] = (
#     "AIDC-AI/Ovis2-1B",
#     "AIDC-AI/Ovis2-2B",
#     "AIDC-AI/Ovis2-4B",
#     "AIDC-AI/Ovis2-8B",
#     "AIDC-AI/Ovis2-16B",
#     "AIDC-AI/Ovis2-34B",
# )

OVIS2_5_VLLM_MODELS: tuple[str, ...] = (
    "AIDC-AI/Ovis2.5-2B",
    "AIDC-AI/Ovis2.5-9B",
)

BASE_VLLM_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "moonshotai/Kimi-VL-A3B-Instruct",
    "deepseek-ai/deepseek-vl2",
    "openbmb/MiniCPM-o-2_6",
    "zai-org/GLM-4.5V",
)

VLLM_ENV_GROUP_OVERRIDES: dict[str, str] = {
    "mistralai/Pixtral-12B-2409": "pixtral",
}

DEFAULT_VLLM_ENV_GROUP = "vllm_normal"

TRANSFORMERS_ENV_GROUP_OVERRIDES: dict[str, str] = {
    "cyberagent/llava-calm2-siglip": "calm",
    "neulab/Pangea-7B-hf": "sarashina",
    "sbintuitions/sarashina2-vision-8b": "sarashina",
    "sbintuitions/sarashina2-vision-14b": "sarashina",
    "microsoft/Phi-4-multimodal-instruct": "phi",
    "turing-motors/Heron-NVILA-Lite-1B": "heron_nvila",
    "turing-motors/Heron-NVILA-Lite-2B": "heron_nvila",
    "turing-motors/Heron-NVILA-Lite-15B": "heron_nvila",
    "turing-motors/Heron-NVILA-Lite-33B": "heron_nvila",
    "llm-jp/llm-jp-3-vila-14b": "vilaja",
    # "Efficient-Large-Model/VILA1.5-13b": "vilaja",
    "SakanaAI/Llama-3-EvoVLM-JP-v2": "evovlm",
}

DEFAULT_TRANSFORMERS_ENV_GROUP = "normal"

API_ENV_GROUP_OVERRIDES: dict[str, str] = {
    "gpt-4o-2024-05-13": "normal",
    "gpt-4o-2024-11-20": "normal",
}

DEFAULT_API_ENV_GROUP = "normal"

__all__ = [
    "BASE_VLLM_MODELS",
    "DEFAULT_VLLM_ENV_GROUP",
    "INTERNVL_VLLM_MODELS",
    "OVIS2_5_VLLM_MODELS",
    # "OVIS2_VLLM_MODELS",
    "VLLM_ENV_GROUP_OVERRIDES",
    "TRANSFORMERS_ENV_GROUP_OVERRIDES",
    "DEFAULT_TRANSFORMERS_ENV_GROUP",
    "API_ENV_GROUP_OVERRIDES",
    "DEFAULT_API_ENV_GROUP",
]
