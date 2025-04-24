from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import transformers
import torch
from base_vlm import BaseVLM
from utils import GenerationConfig


class VLM(BaseVLM):
    def __init__(self, model_id: str = "microsoft/Phi-4-multimodal-instruct") -> None:
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="flash_attention_2",
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def generate(
        self,
        images: list[Image.Image] | None,
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        if images is None:
            images = []
        generation_config = transformers.GenerationConfig.from_pretrained(
            self.model_id, "generation_config.json"
        )

        ########################### vision (multi-frame) ################################
        placeholder = ""
        for i in range(len(images)):
            placeholder += f"<|image_{i}|>"

        messages = [
            {"role": "user", "content": placeholder + text},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if images is None:
            images = []
        inputs = self.processor(prompt, images, return_tensors="pt").to(self.device)

        generate_ids = self.model.generate(
            **inputs,
            **gen_kwargs.__dict__,
            generation_config=generation_config,
        )

        # 入力部分を取り除いた生成結果をデコード
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
