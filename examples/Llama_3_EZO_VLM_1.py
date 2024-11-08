import torch
from mantis.models.conversation import Conversation, SeparatorStyle
from mantis.models.mllava import (
    chat_mllava,
    LlavaForConditionalGeneration,
    MLlavaProcessor,
)
from mantis.models.mllava.utils import conv_templates
from base_vlm import BaseVLM
from utils import GenerationConfig

# 1. Set the system prompt
conv_llama_3_elyza = Conversation(
    system="<|start_header_id|>system<|end_header_id|>\n\nあなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。",
    roles=("user", "assistant"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
    sep="<|eot_id|>",
)
conv_templates["llama_3"] = conv_llama_3_elyza


class VLM(BaseVLM):
    model_id = "AXCXEPT/Llama-3-EZO-VLM-1"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map=self.device
        ).eval()
        self.processor = MLlavaProcessor.from_pretrained(
            "TIGER-Lab/Mantis-8B-siglip-llama3"
        )
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def generate(
        self, image, text: str, gen_kwargs: GenerationConfig = GenerationConfig()
    ):
        if isinstance(image, list):
            if "<image>" not in text:
                text = "<image> " * len(image) + "\n" + text
            images = image
        else:
            if "<image>" not in text:
                text = "<image>\n" + text
            images = [image]
        response, history = chat_mllava(
            text, images, self.model, self.processor, **gen_kwargs.__dict__
        )
        return response


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
