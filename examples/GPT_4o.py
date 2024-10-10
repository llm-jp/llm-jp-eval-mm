from openai import AzureOpenAI
import os
from io import BytesIO
import base64
import requests


def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


class VLM:
    model_id: str = "gpt-4o-2024-05-13"

    def __init__(self) -> None:
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    def generate(self, image, text: str, max_new_tokens: int = 256):
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
                model=self.model_id, messages=message, max_tokens=max_new_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Error:", e)
            return "Error"


if __name__ == "__main__":
    import requests
    from PIL import Image

    model = VLM()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    print(model.generate(image, "What is in the image?"))
