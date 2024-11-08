import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from typing import Union
from base_vlm import BaseVLM
from utils import GenerationConfig


class VLM(BaseVLM):
    model_id = "Kendamarron/Llama-3.2-11B-Vision-Instruct-Swallow-8B-Merge"

    def __init__(self) -> None:
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(
        self,
        images: Union[Image.Image, list[Image.Image]],
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ):
        if "<image>" in text:
            text = text.replace("<image>", "")
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
        inputs = self.processor(
            images, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)
        output_ids = self.model.generate(**inputs, **gen_kwargs.__dict__)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        return self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
