import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from typing import Union
from base_vlm import BaseVLM
from utils import GenerationConfig

DEFAULT_IMAGE_TOKEN = "<image>"


class VLM(BaseVLM):
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        self.model.to(self.device)

    def generate(
        self,
        images: Union[Image.Image, list[Image.Image]],
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ):
        if DEFAULT_IMAGE_TOKEN in text:
            text = text.replace(DEFAULT_IMAGE_TOKEN, "")
        num_images = 1
        if isinstance(images, list):
            num_images = len(images)
        content = [{"type": "image"} for _ in range(num_images)]
        content.extend([{"type": "text", "text": text}])
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = self.processor(images=images, text=input_text, return_tensors="pt").to(
            "cuda"
        )

        # autoregressively complete prompt
        output = self.model.generate(**inputs, **gen_kwargs.__dict__)[0]

        generated_text = self.processor.decode(output, skip_special_tokens=True)
        # split [INST] and return the last part
        generated_text = generated_text.split("[/INST]")[-1].strip()
        return generated_text


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
