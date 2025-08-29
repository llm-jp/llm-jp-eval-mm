from vllm import LLM, SamplingParams
from PIL import Image
from utils import GenerationConfig
from base_vlm import BaseVLM
from vllm_registry import VLLMModelRegistry
import torch


class VLLM(BaseVLM):
    def __init__(self, 
        model_id: str,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = None,
        tensor_parallel_size: int = 1,
    ) -> None:
        self.model_id = model_id
        self.registry = VLLMModelRegistry(self.model_id)
        self.processor = self.registry.processor
        self.vllm_loader = self.registry.loader_map[self.model_id]

        engine_config = self.registry.get_engine_config(self.model_id)
        self.engine_args_dict = {
            "model": self.model_id,
            "tensor_parallel_size": tensor_parallel_size,  # number of GPUs of the machine, but 40 should be divisible by tensor_parallel_size
            "gpu_memory_utilization": gpu_memory_utilization,
            "download_dir": "./.cache/vllm",
            **engine_config,
        }
        self.model = LLM(**self.engine_args_dict)

    def generate(
        self,
        images: list[Image.Image] | None,
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        if images is None:
            images = []
        req_data = self.vllm_loader(text, images)
        sampling_params = SamplingParams(
            temperature=gen_kwargs.temperature,
            max_tokens=gen_kwargs.max_new_tokens,
            stop_token_ids=req_data.stop_token_ids,
        )
        outputs = self.model.generate(
            {
                "prompt": req_data.prompt,
                "multi_modal_data": {"image": req_data.image_data},
            },
            sampling_params=sampling_params,
            lora_request=req_data.lora_requests,
        )
        return outputs[0].outputs[0].text

    def batch_generate(
        self,
        images_list: list[list[Image.Image]] | None,
        text_list: list[str],
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> list[str]:
        if images_list is None:
            images_list = [[] for _ in range(len(text_list))]

        assert len(images_list) == len(text_list)

        from tqdm import tqdm

        req_data_list = []

        for text, images in tqdm(zip(text_list, images_list)):
            req_data_list.append(self.vllm_loader(text, images))

        sampling_params = SamplingParams(
            temperature=gen_kwargs.temperature,
            max_tokens=gen_kwargs.max_new_tokens,
        )

        print(f"Generated {len(req_data_list)} requests")

        outputs = self.model.generate(
            [
                {
                    "prompt": req_data.prompt,
                    "multi_modal_data": {"image": req_data.image_data},
                }
                for req_data in req_data_list
            ],
            sampling_params=sampling_params,
        )
        return [output.outputs[0].text for output in outputs]


if __name__ == "__main__":
    print("=== Qwen/Qwen2.5-VL-3B-Instruct ===")
    vllm = VLLM("Qwen/Qwen2.5-VL-3B-Instruct")
    vllm.test_vlm()
    vllm.test_vlm_batch_100()
