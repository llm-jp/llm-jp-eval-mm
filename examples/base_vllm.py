import dataclasses
from typing import Iterable

from PIL import Image
from vllm import EngineArgs, LLM, SamplingParams
from vllm.lora.request import LoRARequest

from eval_mm.models.base_vlm import BaseVLM
from eval_mm.models.generation_config import GenerationConfig
from vllm_registry import ModelRequestData, VLLMModelRegistry


class VLLM(BaseVLM):
    def __init__(
        self,
        model_id: str,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
    ) -> None:
        self.registry = VLLMModelRegistry(model_id)

        ea = self.registry.get_engine_args()

        # Extract non-default fields from EngineArgs to forward all
        # registry-specified settings (enforce_eager, dtype, mm_processor_kwargs,
        # limit_mm_per_prompt, tokenizer_mode, etc.) to the LLM constructor.
        default_ea = EngineArgs(model=ea.model)
        engine_args: dict = {}
        for f in dataclasses.fields(ea):
            value = getattr(ea, f.name)
            default_value = getattr(default_ea, f.name)
            if value != default_value or f.name == "model":
                engine_args[f.name] = value

        # Override with user-provided values
        engine_args["tensor_parallel_size"] = tensor_parallel_size
        engine_args["gpu_memory_utilization"] = gpu_memory_utilization
        if max_model_len is not None:
            engine_args["max_model_len"] = max_model_len

        self.model_id = model_id
        self.model = LLM(**engine_args)

    def generate(
        self,
        images: list[Image.Image] | None,
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        normalized_images = self._normalize_images(images)

        request = self.registry.build_requests([text], [normalized_images])

        sampling_params = self._build_sampling_params(
            gen_kwargs, request.stop_token_ids
        )

        payload = {
            "prompt": request.prompts[0],
            "multi_modal_data": {
                self.registry.modality: self._prepare_mm_payload(normalized_images)
            },
        }

        lora_request = self._resolve_lora_request(request, 1)
        outputs = self.model.generate(
            payload,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        return outputs[0].outputs[0].text

    def batch_generate(
        self,
        images_list: list[list[Image.Image]] | None,
        text_list: list[str],
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> list[str]:
        if images_list is None:
            images_list = [[] for _ in range(len(text_list))]

        normalized_list = [self._normalize_images(images) for images in images_list]
        request = self.registry.build_requests(text_list, normalized_list)
        sampling_params = self._build_sampling_params(
            gen_kwargs, request.stop_token_ids
        )

        inputs = []
        for idx, (prompt, images) in enumerate(zip(request.prompts, normalized_list)):
            multi_modal_data = self._prepare_mm_payload(images)
            request_payload = {
                "prompt": prompt,
                "multi_modal_data": {self.registry.modality: multi_modal_data},
            }
            inputs.append(request_payload)

        lora_request = self._resolve_lora_request(request, len(inputs))
        outputs = self.model.generate(
            inputs,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        return [output.outputs[0].text for output in outputs]

    def _prepare_mm_payload(
        self, images: list[Image.Image]
    ) -> Image.Image | list[Image.Image] | None:
        if not images:
            return None
        if len(images) == 1:
            return images[0]
        return images

    def _normalize_images(
        self, images: Iterable[Image.Image] | None
    ) -> list[Image.Image]:
        if not images:
            return []

        normalized: list[Image.Image] = []
        for image in images:
            if not isinstance(image, Image.Image):  # pragma: no cover - defensive
                msg = "All images must be PIL.Image instances"
                raise TypeError(msg)

            normalized.append(image if image.mode == "RGB" else image.convert("RGB"))

        return normalized

    def _build_sampling_params(
        self,
        gen_kwargs: GenerationConfig,
        stop_token_ids: list[int] | None,
    ) -> SamplingParams:
        return SamplingParams(
            temperature=gen_kwargs.temperature,
            top_p=gen_kwargs.top_p,
            max_tokens=gen_kwargs.max_new_tokens,
            stop_token_ids=stop_token_ids,
        )

    def _resolve_lora_request(
        self,
        request: ModelRequestData,
        num_prompts: int,
    ) -> list[LoRARequest] | None:
        lora_requests = request.lora_requests
        if not lora_requests:
            return None

        if len(lora_requests) == num_prompts:
            return lora_requests

        if len(lora_requests) == 1:
            return lora_requests * num_prompts

        msg = "Unexpected number of LoRA requests for the current batch"
        raise ValueError(msg)


if __name__ == "__main__":
    vllm = VLLM("Qwen/Qwen3-VL-30B-A3B-Instruct")
    vllm.test_vlm()
    vllm.test_vlm_batch_100()
