from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch
from base_vlm import BaseVLM
from utils import GenerationConfig


class VLM(BaseVLM):
    def __init__(self, model_id: str = "google/gemma-3-4b-it") -> None:
        self.model_id = model_id
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="bfloat16", device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(
        self,
        images: list[Image.Image],
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        image_content = []
        for image in images:
            image_content.append({"type": "image", "image": image})

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [*image_content, {"type": "text", "text": text}],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, **gen_kwargs.__dict__)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
