from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from base_vlm import BaseVLM
from utils import GenerationConfig


class VLM(BaseVLM):
    def __init__(self, model_id: str = "sbintuitions/sarashina2-vision-8b") -> None:
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def generate(
        self,
        images: list[Image.Image],
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        message = [{"role": "user", "content": text}]

        text = self.processor.apply_chat_template(message, add_generation_prompt=True)
        # insert <|prefix|><|file|><|suffix|> after <s>
        text = text.replace(
            "<|prefix|><|file|><|suffix|>", "<|prefix|><|file|><|suffix|>" * len(images)
        )
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        stopping_criteria = self.processor.get_stopping_criteria(["\n###"])

        # Inference: Generation of the output
        output_ids = self.model.generate(
            **inputs,
            **gen_kwargs.__dict__,
            stopping_criteria=stopping_criteria,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text[0]


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
