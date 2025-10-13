import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from transformers import AutoTokenizer
from vllm import EngineArgs
from vllm.lora.request import LoRARequest


@dataclass
class ModelRequestData:
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None


INTERNVL_MODELS: tuple[str, ...] = (
    "OpenGVLab/InternVL3-1B",
    "OpenGVLab/InternVL3-2B",
    "OpenGVLab/InternVL3-8B",
    "OpenGVLab/InternVL3-14B",
    "OpenGVLab/InternVL3-38B",
    "OpenGVLab/InternVL3-78B",
)


class VLLMModelRegistry:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.modality = "image"

        registry: dict[
            str,
            tuple[
                Callable[[], EngineArgs],
                Callable[[list[str], list[list[Image.Image]]], ModelRequestData],
            ],
        ] = {
            "Qwen/Qwen3-VL-30B-A3B-Instruct": (
                self._engine_args_qwen3_vl,
                self._load_qwen3_vl,
            ),
            "moonshotai/Kimi-VL-A3B-Instruct": (
                self._engine_args_kimi_vl,
                self._load_kimi_vl,
            ),
            "deepseek-ai/deepseek-vl2": (
                self._engine_args_deepseek_vl2,
                self._load_deepseek_vl2,
            ),
        }

        for internvl_model in INTERNVL_MODELS:
            registry[internvl_model] = (
                self._engine_args_internvl,
                self._load_internvl,
            )

        try:
            self._engine_resolver, self._request_builder = registry[model_id]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError(
                f"Model {model_id} is not registered for VLM inference"
            ) from exc

    def get_engine_args(self) -> EngineArgs:
        """Return the EngineArgs recommended by the registry."""

        return self._engine_resolver()

    def build_requests(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        """Create prompts and optional extras for the provided inputs."""

        return self._request_builder(texts, images_list)

    def _engine_args_qwen3_vl(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=4096,
            max_num_seqs=5,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            },
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_qwen3_vl(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        if len(texts) != len(images_list):
            msg = "texts and images_list must have identical length"
            raise ValueError(msg)

        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            num_images = len(images)
            if num_images > 0:
                placeholder = "".join("<|image_pad|>" for _ in range(num_images))
                vision_block = f"<|vision_start|>{placeholder}<|vision_end|>"
            else:
                vision_block = ""

            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{vision_block}{text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            prompts.append(prompt)

        return ModelRequestData(prompts=prompts)

    def _engine_args_kimi_vl(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=4096,
            trust_remote_code=True,
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_kimi_vl(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        if len(texts) != len(images_list):
            msg = "texts and images_list must have identical length"
            raise ValueError(msg)

        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            num_images = len(images)
            if num_images > 0:
                placeholder = "".join("<|media_pad|>" for _ in range(num_images))
                vision_block = (
                    "<|media_start|>image<|media_content|>"
                    f"{placeholder}<|media_end|>"
                )
            else:
                vision_block = ""

            prompt = (
                "<|im_user|>user<|im_middle|>"
                f"{vision_block}{text}<|im_end|>"
                "<|im_assistant|>assistant<|im_middle|>"
            )
            prompts.append(prompt)

        return ModelRequestData(prompts=prompts)

    def _engine_args_internvl(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            trust_remote_code=True,
            max_model_len=8192,
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_internvl(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        if len(texts) != len(images_list):
            msg = "texts and images_list must have identical length"
            raise ValueError(msg)

        if not hasattr(self, "_internvl_tokenizer"):
            self._internvl_tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

        tokenizer = self._internvl_tokenizer

        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            num_images = len(images)
            if num_images > 0:
                message_content = [{"type": "image"} for _ in range(num_images)]
                if text:
                    message_content.append({"type": "text", "text": text})
            else:
                message_content = [{"type": "text", "text": text}]

            messages = [[{"role": "user", "content": message_content}]]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt[0])

        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [
            token_id
            for token_id in (
                tokenizer.convert_tokens_to_ids(token) for token in stop_tokens
            )
            if token_id is not None
        ]

        return ModelRequestData(prompts=prompts, stop_token_ids=stop_token_ids)

    def _engine_args_deepseek_vl2(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=4096,
            max_num_seqs=2,
            hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
            limit_mm_per_prompt={self.modality: 1},
        )

    def _load_deepseek_vl2(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        if len(texts) != len(images_list):
            msg = "texts and images_list must have identical length"
            raise ValueError(msg)

        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            num_images = len(images)
            if num_images > 0:
                image_placeholders = " ".join("<image>" for _ in range(num_images))
                user_prefix = f"<|User|>: {image_placeholders}\n"
            else:
                user_prefix = "<|User|>:\n"

            prompt = f"{user_prefix}{text}\n\n<|Assistant|>:"
            prompts.append(prompt)

        return ModelRequestData(prompts=prompts)


def _generate_dummy_images(count: int) -> list[Image.Image]:
    """Return placeholder PIL images for prompt-construction tests."""

    return [Image.new("RGB", (1, 1), color=0) for _ in range(count)]


def preview_qwen3_vl_requests(
    texts: list[str], image_counts: list[int]
) -> ModelRequestData:
    """Build prompts for Qwen3-VL using dummy images (testing helper)."""

    if len(texts) != len(image_counts):
        msg = "texts and image_counts must have identical length"
        raise ValueError(msg)

    images_list = [_generate_dummy_images(count) for count in image_counts]
    registry = VLLMModelRegistry("Qwen/Qwen3-VL-30B-A3B-Instruct")
    return registry.build_requests(texts, images_list)


def preview_kimi_vl_requests(
    texts: list[str], image_counts: list[int]
) -> ModelRequestData:
    """Build prompts for Kimi-VL using dummy images (testing helper)."""

    if len(texts) != len(image_counts):
        msg = "texts and image_counts must have identical length"
        raise ValueError(msg)

    images_list = [_generate_dummy_images(count) for count in image_counts]
    registry = VLLMModelRegistry("moonshotai/Kimi-VL-A3B-Instruct")
    return registry.build_requests(texts, images_list)


def preview_internvl_requests(
    texts: list[str], image_counts: list[int]
) -> ModelRequestData:
    """Build prompts for InternVL using dummy images (testing helper)."""

    if len(texts) != len(image_counts):
        msg = "texts and image_counts must have identical length"
        raise ValueError(msg)

    images_list = [_generate_dummy_images(count) for count in image_counts]
    registry = VLLMModelRegistry("OpenGVLab/InternVL3-2B")
    return registry.build_requests(texts, images_list)


def preview_deepseek_vl2_requests(
    texts: list[str], image_counts: list[int]
) -> ModelRequestData:
    """Build prompts for Deepseek-VL2 using dummy images (testing helper)."""

    if len(texts) != len(image_counts):
        msg = "texts and image_counts must have identical length"
        raise ValueError(msg)

    images_list = [_generate_dummy_images(count) for count in image_counts]
    registry = VLLMModelRegistry("deepseek-ai/deepseek-vl2")
    return registry.build_requests(texts, images_list)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview prompts generated by the VLLM model registry.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        choices=[
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "moonshotai/Kimi-VL-A3B-Instruct",
            "deepseek-ai/deepseek-vl2",
            *INTERNVL_MODELS,
        ],
        help="Registered model identifier to preview.",
    )
    parser.add_argument(
        "--texts",
        default=["What is in the image?"],
        nargs="+",
        help="One or more user messages to build prompts for.",
    )
    parser.add_argument(
        "--image-counts",
        nargs="+",
        type=int,
        help="Number of images to attach per message (broadcast if a single value).",
    )
    parser.add_argument(
        "--show-engine-args",
        action="store_true",
        help="Print the EngineArgs associated with the selected model.",
    )
    return parser.parse_args()


def _broadcast_counts(texts: list[str], image_counts: Optional[list[int]]) -> list[int]:
    if image_counts is None:
        return [0] * len(texts)

    if len(image_counts) == 1 and len(texts) > 1:
        image_counts = image_counts * len(texts)

    if len(image_counts) != len(texts):
        msg = "image_counts must match the number of texts"
        raise ValueError(msg)

    if any(count < 0 for count in image_counts):
        msg = "image_counts must be non-negative"
        raise ValueError(msg)

    return image_counts


def _preview_cli() -> None:
    args = _parse_cli_args()
    texts = args.texts
    image_counts = _broadcast_counts(texts, args.image_counts)

    preview_dispatch: dict[str, Callable[[list[str], list[int]], ModelRequestData]] = {
        "Qwen/Qwen3-VL-30B-A3B-Instruct": preview_qwen3_vl_requests,
        "moonshotai/Kimi-VL-A3B-Instruct": preview_kimi_vl_requests,
        "deepseek-ai/deepseek-vl2": preview_deepseek_vl2_requests,
    }

    for internvl_model in INTERNVL_MODELS:
        preview_dispatch[internvl_model] = preview_internvl_requests

    preview_fn = preview_dispatch[args.model_id]
    registry = VLLMModelRegistry(args.model_id)
    request_data = preview_fn(texts, image_counts)

    if args.show_engine_args:
        engine_args = registry.get_engine_args()
        print("EngineArgs:")
        print(engine_args)
        print()

    for idx, prompt in enumerate(request_data.prompts):
        print(f"Prompt[{idx}]:")
        print(prompt)
        print("---")

    stop_ids = request_data.stop_token_ids
    print("Stop token IDs:", stop_ids if stop_ids is not None else "None")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    _preview_cli()
