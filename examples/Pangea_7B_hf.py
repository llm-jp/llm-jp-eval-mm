# Assuming that you have text_input and image_path
from transformers import LlavaNextForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import requests
from typing import Union


class VLM:
    model_id: str = "neulab/Pangea-7B-hf"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16
        ).to(0)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

    def generate(
        self,
        images: Union[Image.Image, list[Image.Image]],
        text: str,
        max_new_tokens: int = 256,
    ):
        if isinstance(images, list):
            prompt_template = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
                + "\n<image>" * len(images)
                + "\n{text}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            prompt_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{text}<|im_end|>\n<|im_start|>assistant\n"
        if "<image>" in text:
            text = text.replace("<image>", "")
        input_text = prompt_template.format(text=text)
        model_inputs = self.processor(
            images=images, text=input_text, return_tensors="pt"
        ).to("cuda", torch.float16)
        output = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=32,
            temperature=1.0,
            top_p=0.9,
            do_sample=True,
        )
        output = output[0]
        result = self.processor.decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # extract the answer
        result = result.split("assistant\n")[-1].strip()
        return result


if __name__ == "__main__":
    model = VLM()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    print(model.generate(image, "What is in the image?"))

    multi_images = [image for _ in range(3)]
    print(model.generate(multi_images, "What is the difference between these images?"))
