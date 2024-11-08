from openai import AzureOpenAI
import os
from io import BytesIO
import base64
from base_vlm import BaseVLM
from utils import GenerationConfig


def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


class VLM(BaseVLM):
    model_id: str = "gpt-4o-2024-05-13"

    def __init__(self) -> None:
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    def generate(
        self, image, text: str, gen_kwargs: GenerationConfig = GenerationConfig()
    ):
        if "<image>" in text:
            text = text.replace("<image>", "")
        message = []
        if isinstance(image, list):
            image__base64_list = [encode_image_to_base64(img) for img in image]
            message_base = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    },
                ],
            }
            for image_base64 in image__base64_list:
                message_base["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low",
                        },
                    }
                )
            message = [message_base]
        else:
            image_base64 = encode_image_to_base64(image)
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=message,
                max_tokens=gen_kwargs.max_new_tokens,
                temperature=gen_kwargs.temperature,
                top_p=gen_kwargs.top_p,
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Error:", e)
            return "Error"


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
