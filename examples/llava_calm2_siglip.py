from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from base_vlm import BaseVLM
from utils import GenerationConfig


class VLM(BaseVLM):
    model_id = "cyberagent/llava-calm2-siglip"

    def __init__(self) -> None:
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(
        self, image, text: str, gen_kwargs: GenerationConfig = GenerationConfig()
    ):
        prefix = None
        if "<image>" in text:
            prompt = "USER: " + text + "\nASSISTANT: "
        else:
            if isinstance(image, list):
                num_images = len(image)
                prefix = "<image> " * num_images
            else:
                prefix = "<image> "
            prompt = "USER: " + prefix + text + "\nASSISTANT: "

        inputs = (
            self.processor(text=prompt, images=image, return_tensors="pt")
            .to(self.model.device)
            .to(self.model.dtype)
        )
        output_ids = self.model.generate(
            **inputs,
            **gen_kwargs.__dict__,
        )
        generate_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output = self.processor.tokenizer.decode(
            generate_ids[0][:-1], clean_up_tokenization_spaces=False
        )

        return output


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
