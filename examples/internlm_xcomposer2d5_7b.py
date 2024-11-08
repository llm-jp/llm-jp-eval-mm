# flash-attn is required to run this example.
#
import torch
from transformers import AutoModel, AutoTokenizer
import os
from utils import GenerationConfig
from base_vlm import BaseVLM

torch.set_grad_enabled(False)


class VLM(BaseVLM):
    model_id = "internlm/internlm-xcomposer2d5-7b"

    def __init__(self) -> None:
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
        self, image, text: str, gen_kwargs: GenerationConfig = GenerationConfig()
    ):
        text = text.replace("<image>", "")
        if "<image>" not in text:
            if isinstance(image, list):
                image_tokens = "".join(
                    [f"Image{i} <ImageHere>; " for i in range(1, len(image) + 1)]
                )
                text = f"{image_tokens}{text}"
            else:
                text = f"Image1 <ImageHere>; {text}"

        # make tmp files
        os.makedirs("tmp", exist_ok=True)
        image_files = []
        if isinstance(image, list):
            for i, img in enumerate(image):
                file_path = f"tmp/image_{i}.jpg"
                img.save(file_path)
                image_files.append(file_path)
        else:
            file_path = "tmp/image_0.jpg"
            image.save(file_path)
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
