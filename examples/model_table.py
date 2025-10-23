from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, Iterator, Literal

from examples.model_registry_data import (
    BASE_VLLM_MODELS,
    DEFAULT_API_ENV_GROUP,
    DEFAULT_TRANSFORMERS_ENV_GROUP,
    DEFAULT_VLLM_ENV_GROUP,
    API_ENV_GROUP_OVERRIDES,
    INTERNVL_VLLM_MODELS,
    OVIS2_5_VLLM_MODELS,
    # OVIS2_VLLM_MODELS,
    VLLM_ENV_GROUP_OVERRIDES,
    TRANSFORMERS_ENV_GROUP_OVERRIDES,
)

RuntimeType = Literal["transformers", "vllm", "api"]


@dataclass
class RuntimeConfig:
    module_path: str
    env_group: str | None = None


@dataclass
class ModelSpec:
    default_runtime: RuntimeType
    runtimes: dict[RuntimeType, RuntimeConfig]

    def has_runtime(self, runtime: RuntimeType) -> bool:
        return runtime in self.runtimes

    def get_runtime_config(self, runtime: RuntimeType | None = None) -> RuntimeConfig:
        selected = runtime or self.default_runtime
        if selected not in self.runtimes:
            supported = ", ".join(sorted(self.runtimes))
            msg = f"Runtime '{selected}' is not available. Supported runtimes: {supported}"
            raise ValueError(msg)
        return self.runtimes[selected]


MODEL_SPECS: Dict[str, ModelSpec] = {
    "llava-hf/llava-1.5-7b-hf": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.llava_1_5.VLM"
            ),
        },
    ),
    "llava-hf/llava-1.5-13b-hf": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.llava_1_5.VLM"
            ),
        },
    ),
    "llava-hf/llava-v1.6-mistral-7b-hf": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.llava_1_6_mistral_hf.VLM"
            ),
        },
    ),
    "SakanaAI/EvoVLM-JP-v1-7B": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.evovlm_jp_v1.VLM"
            ),
        },
    ),
    "gpt-4o-2024-05-13": ModelSpec(
        default_runtime="api",
        runtimes={
            "api": RuntimeConfig(module_path="examples.runtimes.api.gpt4o.VLM"),
        },
    ),
    "gpt-4o-2024-11-20": ModelSpec(
        default_runtime="api",
        runtimes={
            "api": RuntimeConfig(module_path="examples.runtimes.api.gpt4o.VLM"),
        },
    ),
    "internlm/internlm-xcomposer2d5-7b": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.xcomposer2d5.VLM"
            ),
        },
    ),
    # "OpenGVLab/InternVL2-8B": ModelSpec(
    #     default_runtime="transformers",
    #     runtimes={
    #         "transformers": RuntimeConfig(module_path="examples.runtimes.transformers.internvl2.VLM"),
    #     },
    # ),
    # "OpenGVLab/InternVL2-26B": ModelSpec(
    #     default_runtime="transformers",
    #     runtimes={
    #         "transformers": RuntimeConfig(module_path="examples.runtimes.transformers.internvl2.VLM"),
    #     },
    # ),
    "OpenGVLab/InternVL3-1B": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.internvl3.VLM"
            ),
        },
    ),
    "OpenGVLab/InternVL3-2B": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.internvl3.VLM"
            ),
        },
    ),
    "OpenGVLab/InternVL3-8B": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.internvl3.VLM"
            ),
        },
    ),
    "OpenGVLab/InternVL3-9B": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.internvl3.VLM"
            ),
        },
    ),
    "OpenGVLab/InternVL3-14B": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.internvl3.VLM"
            ),
        },
    ),
    "OpenGVLab/InternVL3-38B": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.internvl3.VLM"
            ),
        },
    ),
    "OpenGVLab/InternVL3-78B": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.internvl3.VLM"
            ),
        },
    ),
    "meta-llama/Llama-3.2-11B-Vision-Instruct": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.llama_3_2_vision.VLM"
            ),
        },
    ),
    "meta-llama/Llama-3.2-90B-Vision-Instruct": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.llama_3_2_vision.VLM"
            ),
        },
    ),
    # "Kendamarron/Llama-3.2-11B-Vision-Instruct-Swallow-8B-Merge": ModelSpec(
    #     default_runtime="transformers",
    #     runtimes={
    #         "transformers": RuntimeConfig(module_path="examples.runtimes.transformers.llama_3_2_vision.VLM"),
    #     },
    # ),
    # "AXCXEPT/Llama-3-EZO-VLM-1": ModelSpec(
    #     default_runtime="transformers",
    #     runtimes={
    #         "transformers": RuntimeConfig(module_path="examples.runtimes.transformers.llama_3_evovlm_jp_v2.VLM"),
    #     },
    # ),
    "SakanaAI/Llama-3-EvoVLM-JP-v2": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.llama_3_evovlm_jp_v2.VLM"
            ),
        },
    ),
    "neulab/Pangea-7B-hf": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.pangea_hf.VLM"
            ),
        },
    ),
    "mistralai/Pixtral-12B-2409": ModelSpec(
        default_runtime="vllm",
        runtimes={
            "vllm": RuntimeConfig(
                module_path="examples.runtimes.vllm.pixtral.VLM",
                env_group=VLLM_ENV_GROUP_OVERRIDES.get("mistralai/Pixtral-12B-2409"),
            ),
        },
    ),
    "Qwen/Qwen2-VL-2B-Instruct": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.qwen2_vl.VLM"
            ),
        },
    ),
    "Qwen/Qwen2-VL-7B-Instruct": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.qwen2_vl.VLM"
            ),
        },
    ),
    "Qwen/Qwen2-VL-72B-Instruct": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.qwen2_vl.VLM"
            ),
        },
    ),
    "Qwen/Qwen2.5-VL-3B-Instruct": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.qwen2_5_vl.VLM"
            ),
        },
    ),
    "Qwen/Qwen2.5-VL-7B-Instruct": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.qwen2_5_vl.VLM"
            ),
        },
    ),
    "Qwen/Qwen2.5-VL-32B-Instruct": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.qwen2_5_vl.VLM"
            ),
        },
    ),
    "Qwen/Qwen2.5-VL-72B-Instruct": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.qwen2_5_vl.VLM"
            ),
        },
    ),
    "llm-jp/llm-jp-3-vila-14b": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.llm_jp_3_vila.VLM"
            ),
        },
    ),
    "stabilityai/japanese-instructblip-alpha": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.japanese_instructblip_alpha.VLM"
            ),
        },
    ),
    "stabilityai/japanese-stable-vlm": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.japanese_stable_vlm.VLM"
            ),
        },
    ),
    "cyberagent/llava-calm2-siglip": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.llava_calm2_siglip.VLM"
            ),
        },
    ),
    # "Efficient-Large-Model/VILA1.5-13b": ModelSpec(
    #     default_runtime="transformers",
    #     runtimes={
    #         "transformers": RuntimeConfig(module_path="vila.VLM"),
    #     },
    # ),
    "google/gemma-3-1b-it": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.gemma3.VLM"
            ),
        },
    ),
    "google/gemma-3-4b-it": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.gemma3.VLM"
            ),
        },
    ),
    "google/gemma-3-12b-it": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.gemma3.VLM"
            ),
        },
    ),
    "google/gemma-3-27b-it": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.gemma3.VLM"
            ),
        },
    ),
    "sbintuitions/sarashina2-vision-8b": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.sarashina2_vision.VLM"
            ),
        },
    ),
    "sbintuitions/sarashina2-vision-14b": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.sarashina2_vision.VLM"
            ),
        },
    ),
    "microsoft/Phi-4-multimodal-instruct": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.phi4_multimodal.VLM"
            ),
        },
    ),
    # "MIL-UT/Asagi-14B": ModelSpec(
    #     default_runtime="transformers",
    #     runtimes={
    #         "transformers": RuntimeConfig(module_path="examples.runtimes.transformers.asagi.VLM"),
    #     },
    # ),
    "turing-motors/Heron-NVILA-Lite-1B": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.heron_nvila.VLM"
            ),
        },
    ),
    "turing-motors/Heron-NVILA-Lite-2B": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.heron_nvila.VLM"
            ),
        },
    ),
    "turing-motors/Heron-NVILA-Lite-15B": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.heron_nvila.VLM"
            ),
        },
    ),
    "turing-motors/Heron-NVILA-Lite-33B": ModelSpec(
        default_runtime="transformers",
        runtimes={
            "transformers": RuntimeConfig(
                module_path="examples.runtimes.transformers.heron_nvila.VLM"
            ),
        },
    ),
    # "CohereLabs/aya-vision-8b": ModelSpec(
    #     default_runtime="transformers",
    #     runtimes={
    #         "transformers": RuntimeConfig(module_path="examples.runtimes.transformers.aya_vision.VLM"),
    #     },
    # ),
    # "CohereLabs/aya-vision-32b": ModelSpec(
    #     default_runtime="transformers",
    #     runtimes={
    #         "transformers": RuntimeConfig(module_path="examples.runtimes.transformers.aya_vision.VLM"),
    #     },
    # ),
}


def _apply_env_group_defaults() -> None:
    for model_id, spec in MODEL_SPECS.items():
        for runtime, config in spec.runtimes.items():
            if config.env_group is not None:
                continue
            if runtime == "transformers":
                config.env_group = TRANSFORMERS_ENV_GROUP_OVERRIDES.get(
                    model_id, DEFAULT_TRANSFORMERS_ENV_GROUP
                )
            elif runtime == "api":
                config.env_group = API_ENV_GROUP_OVERRIDES.get(
                    model_id, DEFAULT_API_ENV_GROUP
                )


def _ensure_vllm_runtime(model_id: str) -> None:
    env_group = VLLM_ENV_GROUP_OVERRIDES.get(model_id, DEFAULT_VLLM_ENV_GROUP)
    config = RuntimeConfig(
        module_path="examples.runtimes.vllm.base.VLLM", env_group=env_group
    )
    spec = MODEL_SPECS.get(model_id)
    if spec is None:
        MODEL_SPECS[model_id] = ModelSpec(
            default_runtime="vllm",
            runtimes={"vllm": config},
        )
        return
    if "vllm" in spec.runtimes:
        return
    spec.runtimes["vllm"] = config


def _populate_vllm_specs() -> None:
    for model_id in BASE_VLLM_MODELS:
        _ensure_vllm_runtime(model_id)
    for model_id in INTERNVL_VLLM_MODELS:
        _ensure_vllm_runtime(model_id)
    # for model_id in OVIS2_VLLM_MODELS:
    #     _ensure_vllm_runtime(model_id)
    # for model_id in OVIS2_5_VLLM_MODELS:
    #     _ensure_vllm_runtime(model_id)


def get_class_from_path(class_path: str):
    """指定されたパスからクラスを動的にインポートして返す"""
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_model_spec(model_id: str) -> ModelSpec:
    return MODEL_SPECS[model_id]


def get_class_from_model_id(model_id: str, runtime: RuntimeType | None = None):
    config = get_model_spec(model_id).get_runtime_config(runtime)
    return get_class_from_path(config.module_path)


def iter_model_specs() -> Iterator[tuple[str, ModelSpec]]:
    return MODEL_SPECS.items()


def get_supported_model_ids(runtime: RuntimeType | None = None) -> tuple[str, ...]:
    if runtime is None:
        return tuple(sorted(MODEL_SPECS))
    return tuple(
        sorted(
            model_id
            for model_id, spec in MODEL_SPECS.items()
            if spec.has_runtime(runtime)
        )
    )


_apply_env_group_defaults()
_populate_vllm_specs()


if __name__ == "__main__":
    for model_id, spec in sorted(iter_model_specs()):
        runtimes = ", ".join(sorted(spec.runtimes))
        print(f"{model_id}: default={spec.default_runtime} runtimes=[{runtimes}]")
