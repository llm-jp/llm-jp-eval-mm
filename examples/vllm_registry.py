from typing import Optional
from dataclasses import dataclass
from PIL import Image
from vllm.lora.request import LoRARequest
from transformers import AutoProcessor


@dataclass
class ModelRequestData:
    prompt: str
    image_data: Optional[list[Image.Image]]
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None


class VLLMModelRegistry:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.loader_map = {
            "Qwen/Qwen2.5-VL-3B-Instruct": self.load_qwen2_5_vl,
            "google/gemma-3-4b-it": self.load_gemma3,
        }

    def get_engine_config(self, model_id: str) -> dict:
        return {
            "max_model_len": 8192,
            "max_num_seqs": 5,
            "limit_mm_per_prompt": {"image": 5},
            "trust_remote_code": True,
        }

    def load_qwen2_5_vl(
        self, text: str, images: list[Image.Image] | None
    ) -> ModelRequestData:
        try:
            from qwen_vl_utils import process_vision_info
        except ModuleNotFoundError:
            print(
                "WARNING: `qwen-vl-utils` not installed, input images will not "
                "be automatically resized. You can enable this functionality by "
                "`pip install qwen-vl-utils`."
            )
            process_vision_info = None

        if images is None:
            images = []

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

        if process_vision_info is None:
            image_data = images
        else:
            image_data, _ = process_vision_info(messages, return_video_kwargs=False)

        return ModelRequestData(
            prompt=prompt,
            image_data=image_data,
        )

    def load_gemma3(
        self, text: str, images: list[Image.Image] | None
    ) -> ModelRequestData:
        if images is None:
            images = []

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
            prompt=prompt,
            image_data=images,
        )
