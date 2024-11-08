from PIL import Image
from vllm import LLM
from vllm.sampling_params import SamplingParams
from typing import Union
import base64
from io import BytesIO
from base_vlm import BaseVLM
from utils import GenerationConfig


def image_to_base64(img):
    buffer = BytesIO()
    # Check if the image has an alpha channel (RGBA)
    if img.mode == "RGBA":
        # Convert the image to RGB mode
        img = img.convert("RGB")
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode("ascii")
    return img_str


def image_to_content(image: Image.Image) -> dict:
    base64_image = image_to_base64(image)
    content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
    }
    return content


class VLM(BaseVLM):
    model_id = "mistralai/Pixtral-12B-2409"

    def __init__(self) -> None:
        self.model = LLM(
            model=self.model_id,
            tokenizer_mode="mistral",
            tensor_parallel_size=2,
        )

    def generate(
        self,
        images: Union[Image.Image, list[Image.Image]],
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ):
        if isinstance(images, list):
            content = [image_to_content(image) for image in images]
        else:
            content = [image_to_content(images)]

        content.extend([{"type": "text", "text": text}])
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        sampling_params = SamplingParams(
            max_tokens=gen_kwargs.max_new_tokens,
            temperature=gen_kwargs.temperature,
            top_p=gen_kwargs.top_p,
        )
        outputs = self.model.chat(
            messages,
            sampling_params=sampling_params,
        )
        return outputs[0].outputs[0].text


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
