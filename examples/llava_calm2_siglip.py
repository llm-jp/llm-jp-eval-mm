from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from base_vlm import BaseVLM
from utils import GenerationConfig


class VLM(BaseVLM):
    def __init__(self, model_id: str = "cyberagent/llava-calm2-siglip") -> None:
        self.model_id = model_id
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(
        self, images, text: str, gen_kwargs: GenerationConfig = GenerationConfig()
    ) -> str:
        prefix = None
        if "<image>" in text:
            prompt = "USER: " + text + "\nASSISTANT: "
        else:
            num_images = len(images)
            prefix = "<image> " * num_images
            prompt = "USER: " + prefix + text + "\nASSISTANT: "

        inputs = (
            self.processor(text=prompt, images=images, return_tensors="pt")
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
