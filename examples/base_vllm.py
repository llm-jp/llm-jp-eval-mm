from vllm import LLM, SamplingParams
from typing import Dict, Callable
from PIL import Image
from utils import GenerationConfig
from base_vlm import BaseVLM
from examples.vllm_registry import ModelRequestData, VLLMModelRegistry

class VLLM(BaseVLM):
    MODEL_LOADERS: Dict[str, Callable[[str, int], ModelRequestData]] = {
        "Qwen/Qwen2.5-VL-3B-Instruct": VLLMModelRegistry.load_qwen2_5_vl,
        "google/gemma-3-4b-it": VLLMModelRegistry.load_gemma3,
    }

    def __init__(self, model_id: str = "google/gemma-3-4b-it") -> None:
        self.model_id = model_id
        self.model = LLM(model=self.model_id)

    def generate(
        self, 
        images: list[Image.Image] | None,
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:

        if images is None:
            images = []
        req_data = self.MODEL_LOADERS[self.model_id](text, images)
        sampling_params = SamplingParams(temperature=0.0,
                                        max_tokens=256,
                                        stop_token_ids=req_data.stop_token_ids)
        outputs = self.model.chat(
            req_data.prompt,
            sampling_params=sampling_params,
        )
        return outputs[0].outputs[0].text


if __name__ == "__main__":
    vllm = VLLM("google/gemma-3-4b-it")
    vllm.test_vlm()
    vllm.test_vlm_1000()
