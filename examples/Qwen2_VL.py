from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from base_vlm import BaseVLM
from utils import GenerationConfig


class VLM(BaseVLM):
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct") -> None:
        self.model_id = model_id

        if "Qwen2.5-VL" in model_id:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype="bfloat16",
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        elif "Qwen2-VL" in model_id:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype="bfloat16",
                device_map="auto",
                attn_implementation="flash_attention_2",
            )

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def generate(
        self, images, text: str, gen_kwargs: GenerationConfig = GenerationConfig()
    ) -> str:
        if "<image>" in text:
            text = text.replace("<image>", "")
        message = []
        image_content = []

        for img in images:
            image_content.append(
                {
                    "type": "image",
                    "image": img,
                }
            )
        message.append(
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": text}],
            }
        )

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
