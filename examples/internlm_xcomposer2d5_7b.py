# flash-attn is required to run this example.
#
import torch
from transformers import AutoModel, AutoTokenizer
import os
from PIL import Image

torch.set_grad_enabled(False)


class VLM:
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

    def generate(self, image, text: str, max_new_tokens: int = 256):
        text = text.replace("<image>", "")
        if "<image>" not in text:
            if isinstance(image, list):
                image_tokens = "".join(
                    [f"Image{i} <ImageHere>; " for i in range(len(image))]
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
                do_sample=False,
                num_beams=3,
                use_meta=True,
            )

        # remove tmp files
        for file_path in image_files:
            os.remove(file_path)
        return response


if __name__ == "__main__":
    import requests
    from PIL import Image

    model = VLM()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    print(model.generate(image, "What is in the image?"))
    print(
        model.generate([image, image], "What is the difference between these images?")
    )
