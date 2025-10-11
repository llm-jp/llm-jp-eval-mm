from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from vllm import EngineArgs
from vllm.lora.request import LoRARequest


@dataclass
class ModelRequestData:
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None


class VLLMModelRegistry:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.modality = "image"

        registry: dict[
            str,
            tuple[
                Callable[[], EngineArgs],
                Callable[[list[str], list[list[Image.Image]]], ModelRequestData],
            ],
        ] = {
            "Qwen/Qwen3-VL-30B-A3B-Instruct": (
                self._engine_args_qwen3_vl,
                self._load_qwen3_vl,
            ),
        }

        try:
            self._engine_resolver, self._request_builder = registry[model_id]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError(
                f"Model {model_id} is not registered for VLM inference"
            ) from exc

    def get_engine_args(self) -> EngineArgs:
        """Return the EngineArgs recommended by the registry."""

        return self._engine_resolver()

    def build_requests(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        """Create prompts and optional extras for the provided inputs."""

        return self._request_builder(texts, images_list)

    def _engine_args_qwen3_vl(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=4096,
            max_num_seqs=5,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            },
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_qwen3_vl(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        if len(texts) != len(images_list):
            msg = "texts and images_list must have identical length"
            raise ValueError(msg)

        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            num_images = len(images)
            if num_images > 0:
                placeholder = "".join("<|image_pad|>" for _ in range(num_images))
                vision_block = f"<|vision_start|>{placeholder}<|vision_end|>"
            else:
                vision_block = ""

            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{vision_block}{text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            prompts.append(prompt)

        return ModelRequestData(prompts=prompts)
