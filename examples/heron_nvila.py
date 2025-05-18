from base_vlm import BaseVLM
from utils import GenerationConfig
import torch
from transformers import GenerationConfig as HFGenerationConfig, AutoModel


def create_prompt(text, image):
    if image is None or (isinstance(image, list) and len(image) == 0):
        return [text] if text else []
    if not isinstance(image, list):
        image = [image]
    if not text:
        return image
    if "<image>" not in text:
        prompt = image.copy()
        prompt.append(text)
        return prompt
    parts = text.split("<image>")
    prompt, idx = [], 0
    if parts[0] == "":
        prompt.append(image[idx])
        idx += 1
        parts = parts[1:]
    for i, part in enumerate(parts):
        if part:
            prompt.append(part)
        if idx < len(image) and (i < len(parts) - 1 or text.endswith("<image>")):
            prompt.append(image[idx])
            idx += 1
    return prompt


class VLM(BaseVLM):
    def __init__(self, model_id="turing-motors/Heron-NVILA-Lite-15B"):
        self.model_id = model_id
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, device_map="auto"
        )

    def generate(
        self, image, text: str, gen_kwargs: GenerationConfig = GenerationConfig()
    ):
        gen_cfg = HFGenerationConfig(**gen_kwargs.__dict__)
        prompt = create_prompt(text, image)
        with torch.no_grad():
            return self.model.generate_content(prompt, generation_config=gen_cfg)


if __name__ == "__main__":
    VLM("turing-motors/Heron-NVILA-Lite-15B").test_vlm()
