from openai import AzureOpenAI, APIError
import os
from io import BytesIO
import base64
from base_vlm import BaseVLM
from utils import GenerationConfig
import backoff
from PIL import Image


def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


@backoff.on_exception(backoff.expo, (ValueError, APIError), max_tries=5)
def make_api_call(vlm, message, gen_kwargs):
    return vlm.client.chat.completions.create(
        model=vlm.model_id,
        messages=message,
        max_tokens=gen_kwargs.max_new_tokens,
        temperature=gen_kwargs.temperature,
        top_p=gen_kwargs.top_p,
    )


class VLM(BaseVLM):
    def __init__(self, model_id: str = "gpt-4o-2024-05-13") -> None:
        self.model_id = model_id
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    def generate(
        self,
        images: list[Image.Image] | None,
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        message = []
        content: list[dict[str, str | dict[str, str]]] = []
        if images is None:
            images = []
        image_base64_list = [encode_image_to_base64(img) for img in images]
        message_base = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text,
                },
            ],
        }

        content.append(
            {
                "type": "text",
                "text": text,
            },
        )
        for image_base64 in image_base64_list:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "auto",
                    },
                }
            )
        message_base = {
            "role": "user",
            "content": content,
        }
        message = [message_base]

        response = make_api_call(self, message, gen_kwargs)
        return response.choices[0].message.content


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
