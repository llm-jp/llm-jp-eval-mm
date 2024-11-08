from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from io import BytesIO
import base64
from qwen_vl_utils import process_vision_info
from base_vlm import BaseVLM
from utils import GenerationConfig


class VLM(BaseVLM):
    model_id = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(self) -> None:
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="bfloat16", device_map="auto"
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def generate(
        self, image, text: str, gen_kwargs: GenerationConfig = GenerationConfig()
    ):
        if "<image>" in text:
            text = text.replace("<image>", "")
        message = []
        if isinstance(image, list):
            image_content = []

            for img in image:
                base64_image = img.convert("RGB")
                buffer = BytesIO()
                base64_image.save(buffer, format="JPEG")
                base64_bytes = base64.b64encode(buffer.getvalue())
                base64_string = base64_bytes.decode("utf-8")
                image_content.append(
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{base64_string}",
                    }
                )
            message.append(
                {
                    "role": "user",
                    "content": image_content + [{"type": "text", "text": text}],
                }
            )
        else:
            base64_image = image.convert("RGB")
            buffer = BytesIO()
            base64_image.save(buffer, format="JPEG")
            base64_bytes = base64.b64encode(buffer.getvalue())
            base64_string = base64_bytes.decode("utf-8")
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{base64_string}",
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ]

        texts = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[texts],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)
        output_ids = self.model.generate(**inputs, **gen_kwargs.__dict__)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return generated_text


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
