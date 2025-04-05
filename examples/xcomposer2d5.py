# flash-attn is required to run this example.
#
import torch
from transformers import AutoModel, AutoTokenizer
import os
from utils import GenerationConfig
from base_vlm import BaseVLM

torch.set_grad_enabled(False)


class VLM(BaseVLM):
    def __init__(self, model_id: str = "internlm/internlm-xcomposer2d5-7b") -> None:
        self.model_id = model_id
        self.model = (
            AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            .cuda()
            .eval()
            .half()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model.tokenizer = self.tokenizer

    def generate(
        self, images, text: str, gen_kwargs: GenerationConfig = GenerationConfig()
    ) -> str:
        if "<image>" not in text:
            image_tokens = "".join(
                [f"Image{i} <ImageHere>; " for i in range(1, len(images) + 1)]
            )
            text = f"{image_tokens}{text}"
        # make tmp files
        os.makedirs("tmp", exist_ok=True)
        image_files = []
        for i, img in enumerate(images):
            file_path = f"tmp/image_{i}.jpg"
            img.save(file_path)
            image_files.append(file_path)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            response, _ = self.model.chat(
                self.tokenizer,
                text,
                image_files,
                generation_config=gen_kwargs.__dict__,
            )

        # remove tmp files
        for file_path in image_files:
            os.remove(file_path)
        return response


if __name__ == "__main__":
    model = VLM()
    model.test_vlm()
