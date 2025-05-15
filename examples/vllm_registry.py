from typing import Optional
from dataclasses import dataclass
from PIL import Image
from vllm import EngineArgs
from vllm.lora.request import LoRARequest
from transformers import AutoProcessor


@dataclass
class ModelRequestData:
    engine_args: EngineArgs
    prompt: list[dict]
    image_data: list[Image.Image]
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None


class VLLMModelRegistry:
    """VLLM Model registry for different models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.loader_map = {
            "Qwen/Qwen2.5-VL-3B-Instruct": self.load_qwen2_5_vl,
            "google/gemma-3-4b-it": self.load_gemma3,
        }

    def get_engine_args(self, model_name: str) -> EngineArgs:
        return self.loader_map[model_name](model_name, "", None).engine_args

    # Loader functions for each model type, similar to the example
    def load_qwen2_5_vl(
        self, model_name: str, text: str, images: list[Image.Image] | None
    ) -> ModelRequestData:
        if images is None:
            images = []

        engine_args = EngineArgs(
            model=model_name,
            max_model_len=4096,
            max_num_seqs=5,
            limit_mm_per_prompt={"image": 5},
        )

        placeholders = [{"type": "image", "image": image} for image in images]
        messages = [
            {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": text},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return ModelRequestData(
            engine_args=engine_args,
            prompt=prompt,
            image_data=images,
        )

    def load_gemma3(
        self, model_name: str, text: str, images: list[Image.Image] | None
    ) -> ModelRequestData:
        if images is None:
            images = []

        engine_args = EngineArgs(
            model=model_name,
            max_model_len=8192,
            max_num_seqs=2,
            limit_mm_per_prompt={"image": 5},
        )

        placeholders = [{"type": "image", "image": image} for image in images]
        messages = [
            {
                "role": "user",
                "content": [
                    *placeholders,
                    {"type": "text", "text": text},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return ModelRequestData(
            engine_args=engine_args,
            prompt=prompt,
            image_data=images,
        )
