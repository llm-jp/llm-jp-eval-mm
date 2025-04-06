import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from base_vlm import BaseVLM
from utils import GenerationConfig


class VLM(BaseVLM):
    def __init__(self, model_id: str = "MIL-UT/Asagi-14B") -> None:
        self.model_id = model_id
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(
        self,
        images: list[Image.Image] | None,
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        if images is None:
            images = []

        prompt = f"""以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。
        ### 指示:
        {"<image>"*len(images)}
        {text}
        ### 応答:
        """

        inputs = self.processor(text=prompt, images=images, return_tensors="pt")

        inputs_text = self.processor.tokenizer(prompt, return_tensors="pt")
        inputs["input_ids"] = inputs_text["input_ids"]
        inputs["attention_mask"] = inputs_text["attention_mask"]
        inputs = {
            k: inputs[k].to(self.model.device) for k in inputs if k != "token_type_ids"
        }

        generate_ids = self.model.generate(**inputs, **gen_kwargs.__dict__)
        generated_text = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # truncate the text to remove the prompt
        generated_text = generated_text.split("### 応答:")[1].strip()
        return generated_text


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
