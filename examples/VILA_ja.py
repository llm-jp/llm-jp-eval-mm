import torch
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from base_vlm import BaseVLM
from utils import GenerationConfig


class VLM(BaseVLM):
    model_id = "llm-jp/VILA-ja"

    def __init__(self) -> None:
        model_checkpoint_path = "/model/sasagawa/VILA-ja/checkpoints/llm-jp-3-13b-instruct_siglip_mlp2xgelu_step-2_20241004/"
        model_name = get_model_name_from_path(model_checkpoint_path)
        print(model_name)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(model_checkpoint_path, model_name)
        )

    def generate(
        self, image, text: str, gen_kwargs: GenerationConfig = GenerationConfig()
    ):
        qs = text
        qs = qs.replace("<image>", "")
        if "<image>" not in text:
            if isinstance(image, list):
                qs = "<image>\n" * len(image) + text
            else:
                qs = "<image>\n" + text
        conv_mode = "llmjp_v3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if isinstance(image, list):
            images = image
        else:
            images = [image]
        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[
                    images_tensor,
                ],
                **gen_kwargs.__dict__,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return outputs


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
