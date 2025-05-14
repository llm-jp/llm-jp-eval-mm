from typing import Optional
from dataclasses import dataclass
from PIL import Image
from io import BytesIO
from vllm import EngineArgs
from vllm.lora.request import LoRARequest
import base64


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

@dataclass
class ModelRequestData:
    engine_args: EngineArgs
    prompt: list[dict]
    image_data: list[Image.Image]
    stop_token_ids: Optional[list[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[list[LoRARequest]] = None

class VLLMModelRegistry:
    """VLLM Model registry for different models."""

    # Loader functions for each model type, similar to the example
    @staticmethod
    def load_qwen2_5_vl(text: str, images: list[Image.Image]|None) -> ModelRequestData:
        try:
            from qwen_vl_utils import process_vision_info
        except ModuleNotFoundError:
            print('WARNING: `qwen-vl-utils` not installed, input images will not '
                'be automatically resized. You can enable this functionality by '
                '`pip install qwen-vl-utils`.')
            process_vision_info = None

        if images is None:
            images = []

        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

        engine_args = EngineArgs(
            model=model_name,
            max_model_len=32768 if process_vision_info is None else 4096,
            max_num_seqs=5,
            limit_mm_per_prompt={"image": len(images)},
        )

        content = [image_to_content(image) for image in images]
        content.extend([{"type": "text", "text": text}])

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            }, 
            {
                "role": "user",
                "content": content,
            }
        ]

        return ModelRequestData(
            engine_args=engine_args,
            prompt=messages,
            image_data=images,
        )
    
    def load_gemma3(text: str, images: list[Image.Image]|None) -> ModelRequestData:
        model_name = "google/gemma-3-4b-it"

        if images is None:
            images = []

        engine_args = EngineArgs(
            model=model_name,
            max_model_len=8192,
            max_num_seqs=2,
            limit_mm_per_prompt={"image": len(images)},
        )

        content = [image_to_content(image) for image in images]
        content.extend([{"type": "text", "text": text}])

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            }, 
            {
                "role": "user",
                "content": content,
            }   
        ]

        return ModelRequestData(
            engine_args=engine_args,
            prompt=messages,
            image_data=images,
        )




