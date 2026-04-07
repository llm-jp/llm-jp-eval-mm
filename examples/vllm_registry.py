import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from vllm import EngineArgs
from vllm.lora.request import LoRARequest


@dataclass
class ModelRequestData:
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None


# ── Model groups ──────────────────────────────────────────────────

LLAVA_1_5_MODELS: tuple[str, ...] = (
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
)

LLAVA_NEXT_MODELS: tuple[str, ...] = (
    "llava-hf/llava-v1.6-mistral-7b-hf",
)

PANGEA_MODELS: tuple[str, ...] = (
    "neulab/Pangea-7B-hf",
)

QWEN2_VL_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct",
)

QWEN2_5_VL_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
)

INTERNVL2_MODELS: tuple[str, ...] = (
    "OpenGVLab/InternVL2-8B",
    "OpenGVLab/InternVL2-26B",
)

INTERNVL3_MODELS: tuple[str, ...] = (
    "OpenGVLab/InternVL3-1B",
    "OpenGVLab/InternVL3-2B",
    "OpenGVLab/InternVL3-8B",
    "OpenGVLab/InternVL3-9B",
    "OpenGVLab/InternVL3-14B",
    "OpenGVLab/InternVL3-38B",
    "OpenGVLab/InternVL3-78B",
)

# Backward compat alias
INTERNVL_MODELS = INTERNVL3_MODELS

GEMMA3_MODELS: tuple[str, ...] = (
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
)

AYA_VISION_MODELS: tuple[str, ...] = (
    "CohereLabs/aya-vision-8b",
    "CohereLabs/aya-vision-32b",
)

OVIS2_MODELS: tuple[str, ...] = (
    "AIDC-AI/Ovis2-1B",
    "AIDC-AI/Ovis2-2B",
    "AIDC-AI/Ovis2-4B",
    "AIDC-AI/Ovis2-8B",
    "AIDC-AI/Ovis2-16B",
    "AIDC-AI/Ovis2-34B",
)

OVIS2_5_MODELS: tuple[str, ...] = (
    "AIDC-AI/Ovis2.5-2B",
    "AIDC-AI/Ovis2.5-9B",
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
            # ── Standalone models ─────────────────────────────
            "Qwen/Qwen3-VL-30B-A3B-Instruct": (
                self._engine_args_qwen3_vl,
                self._load_qwen_vl,
            ),
            "moonshotai/Kimi-VL-A3B-Instruct": (
                self._engine_args_kimi_vl,
                self._load_kimi_vl,
            ),
            "deepseek-ai/deepseek-vl2": (
                self._engine_args_deepseek_vl2,
                self._load_deepseek_vl2,
            ),
            "openbmb/MiniCPM-o-2_6": (
                self._engine_args_minicpm_o,
                self._load_minicpm_o,
            ),
            "zai-org/GLM-4.5V": (
                self._engine_args_glm4_5v,
                self._load_glm4_5v,
            ),
            "microsoft/Phi-4-multimodal-instruct": (
                self._engine_args_phi4,
                self._load_phi4,
            ),
        }

        # ── Model-group registrations ─────────────────────────
        for m in LLAVA_1_5_MODELS:
            registry[m] = (self._engine_args_llava_1_5, self._load_llava_1_5)

        for m in LLAVA_NEXT_MODELS:
            registry[m] = (self._engine_args_llava_next, self._load_llava_next)

        for m in PANGEA_MODELS:
            registry[m] = (self._engine_args_pangea, self._load_pangea)

        for m in QWEN2_VL_MODELS:
            registry[m] = (self._engine_args_qwen2_vl, self._load_qwen_vl)

        for m in QWEN2_5_VL_MODELS:
            registry[m] = (self._engine_args_qwen2_5_vl, self._load_qwen_vl)

        for m in INTERNVL2_MODELS:
            registry[m] = (self._engine_args_internvl, self._load_internvl2)

        for m in INTERNVL3_MODELS:
            registry[m] = (self._engine_args_internvl, self._load_internvl3)

        for m in GEMMA3_MODELS:
            registry[m] = (self._engine_args_gemma3, self._load_gemma3)

        for m in AYA_VISION_MODELS:
            registry[m] = (self._engine_args_aya_vision, self._load_aya_vision)

        for m in OVIS2_MODELS:
            registry[m] = (self._engine_args_ovis2, self._load_ovis2)

        for m in OVIS2_5_MODELS:
            registry[m] = (self._engine_args_ovis2_5, self._load_ovis2_5)

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

    # ── Helper ────────────────────────────────────────────────────

    def _validate_lengths(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> None:
        if len(texts) != len(images_list):
            msg = "texts and images_list must have identical length"
            raise ValueError(msg)

    # ── LLaVA 1.5 ────────────────────────────────────────────────

    def _engine_args_llava_1_5(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=4096,
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_llava_1_5(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)
        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            placeholders = "\n".join("<image>" for _ in range(len(images)))
            if placeholders:
                prompt = f"USER: {placeholders}\n{text}\nASSISTANT:"
            else:
                prompt = f"USER: {text}\nASSISTANT:"
            prompts.append(prompt)
        return ModelRequestData(prompts=prompts)

    # ── LLaVA-NeXT (1.6 Mistral) ─────────────────────────────────

    def _engine_args_llava_next(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=8192,
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_llava_next(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)
        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            placeholders = "".join("<image>" for _ in range(len(images)))
            prompt = f"[INST] {placeholders}\n{text} [/INST]"
            prompts.append(prompt)
        return ModelRequestData(prompts=prompts)

    # ── Pangea (LLaVA-NeXT architecture, Qwen-style template) ────

    def _engine_args_pangea(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=4096,
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_pangea(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)
        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            image_section = "".join("\n<image>" for _ in range(len(images)))
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user{image_section}\n{text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            prompts.append(prompt)
        return ModelRequestData(prompts=prompts)

    # ── Qwen VL family (shared prompt builder) ────────────────────

    def _engine_args_qwen2_vl(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=32768,
            max_num_seqs=5,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            limit_mm_per_prompt={self.modality: 5},
        )

    def _engine_args_qwen2_5_vl(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=32768,
            max_num_seqs=5,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            limit_mm_per_prompt={self.modality: 5},
        )

    def _engine_args_qwen3_vl(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=32768,
            max_num_seqs=5,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            },
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_qwen_vl(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        """Shared prompt builder for Qwen2-VL, Qwen2.5-VL, and Qwen3-VL."""
        self._validate_lengths(texts, images_list)
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

    # ── Kimi-VL ───────────────────────────────────────────────────

    def _engine_args_kimi_vl(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=8192,
            trust_remote_code=True,
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_kimi_vl(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)
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

    # ── InternVL (2 & 3) ─────────────────────────────────────────

    def _engine_args_internvl(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            trust_remote_code=True,
            max_model_len=32768,
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_internvl2(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        """InternVL2 uses string content with <image> tokens."""
        self._validate_lengths(texts, images_list)

        if not hasattr(self, "_internvl_tokenizer"):
            self._internvl_tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

        tokenizer = self._internvl_tokenizer
        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            image_tokens = " ".join("<image>" for _ in range(len(images)))
            content = f"{image_tokens}\n{text}" if image_tokens else text
            messages = [{"role": "user", "content": content}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [
            token_id
            for token_id in (
                tokenizer.convert_tokens_to_ids(token) for token in stop_tokens
            )
            if token_id is not None
        ]
        return ModelRequestData(prompts=prompts, stop_token_ids=stop_token_ids)

    def _load_internvl3(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        """InternVL3 uses multimodal content format."""
        self._validate_lengths(texts, images_list)

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

    # ── Gemma-3 ───────────────────────────────────────────────────

    def _engine_args_gemma3(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=8192,
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_gemma3(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)

        if not hasattr(self, "_gemma3_processor"):
            self._gemma3_processor = AutoProcessor.from_pretrained(self.model_id)

        processor = self._gemma3_processor
        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            content: list[dict] = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": text})
            messages = [{"role": "user", "content": content}]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
        return ModelRequestData(prompts=prompts)

    # ── Phi-4 multimodal ──────────────────────────────────────────

    def _engine_args_phi4(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_phi4(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)
        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            placeholder = "".join(f"<|image_{i + 1}|>" for i in range(len(images)))
            prompt = f"<|user|>\n{placeholder}{text}<|end|>\n<|assistant|>\n"
            prompts.append(prompt)
        return ModelRequestData(prompts=prompts)

    # ── Aya Vision (Cohere) ───────────────────────────────────────

    def _engine_args_aya_vision(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=8192,
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_aya_vision(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)
        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            image_tokens = "<image>" * len(images)
            prompt = (
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
                f"{image_tokens}{text}"
                "<|END_OF_TURN_TOKEN|>"
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            )
            prompts.append(prompt)
        return ModelRequestData(prompts=prompts)

    # ── Asagi (InternVL architecture, custom Japanese template) ───

    def _engine_args_asagi(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_asagi(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)
        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            image_tokens = "<image>" * len(images)
            prompt = (
                "以下は、タスクを説明する指示です。"
                "要求を適切に満たす応答を書きなさい。\n"
                f"### 指示:\n{image_tokens}\n{text}\n### 応答:\n"
            )
            prompts.append(prompt)
        return ModelRequestData(prompts=prompts)

    # ── DeepSeek-VL2 ──────────────────────────────────────────────

    def _engine_args_deepseek_vl2(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=4096,
            max_num_seqs=2,
            hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_deepseek_vl2(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)
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

    # ── GLM-4.5V ──────────────────────────────────────────────────

    def _engine_args_glm4_5v(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=8192,
            max_num_seqs=2,
            trust_remote_code=True,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 5, "video": 0},
        )

    def _load_glm4_5v(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)

        if not hasattr(self, "_glm_processor"):
            self._glm_processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

        processor = self._glm_processor
        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            placeholders = [{"type": "image", "image": image} for image in images]
            messages = [
                {
                    "role": "user",
                    "content": [
                        *placeholders,
                        {"type": "text", "text": text},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
        return ModelRequestData(prompts=prompts)

    # ── Ovis2 ─────────────────────────────────────────────────────

    def _engine_args_ovis2(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=8192,
            max_num_seqs=2,
            trust_remote_code=True,
            dtype="half",
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_ovis2(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)

        if not hasattr(self, "_ovis_tokenizer"):
            self._ovis_tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

        tokenizer = self._ovis_tokenizer
        messages = []
        for text, images in zip(texts, images_list):
            num_images = len(images)
            placeholder_lines = "\n".join("<image>" for _ in range(num_images))
            if placeholder_lines and text:
                content = f"{placeholder_lines}\n{text}"
            elif placeholder_lines:
                content = placeholder_lines
            else:
                content = text
            messages.append([{"role": "user", "content": content}])

        prompts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return ModelRequestData(prompts=prompts)

    # ── Ovis2.5 ───────────────────────────────────────────────────

    def _engine_args_ovis2_5(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=8192,
            max_num_seqs=2,
            trust_remote_code=True,
            dtype="half",
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_ovis2_5(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)

        placeholder_map = {
            "image": "<image>",
            "video": "<video>",
        }
        placeholder = placeholder_map.get(self.modality, "<image>")

        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            num_images = len(images)
            lines: list[str] = []
            if num_images > 0:
                media_block = "\n".join(placeholder for _ in range(num_images))
                lines.append(media_block)
            if text:
                lines.append(text)

            content_block = "\n".join(lines)
            if content_block:
                content_block = f"{content_block}\n"

            prompt = (
                "<|im_start|>user\n\n"
                f"{content_block}"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            prompts.append(prompt)
        return ModelRequestData(prompts=prompts)

    # ── MiniCPM-o ─────────────────────────────────────────────────

    def _engine_args_minicpm_o(self) -> EngineArgs:
        return EngineArgs(
            model=self.model_id,
            max_model_len=4096,
            max_num_seqs=2,
            trust_remote_code=True,
            limit_mm_per_prompt={self.modality: 5},
        )

    def _load_minicpm_o(
        self, texts: list[str], images_list: list[list[Image.Image]]
    ) -> ModelRequestData:
        self._validate_lengths(texts, images_list)

        if not hasattr(self, "_minicpm_tokenizer"):
            self._minicpm_tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

        tokenizer = self._minicpm_tokenizer

        stop_tokens = ("<|im_end|>", "<|endoftext|>")
        stop_token_ids = [
            token_id
            for token_id in (
                tokenizer.convert_tokens_to_ids(token) for token in stop_tokens
            )
            if token_id is not None
        ]

        prompts: list[str] = []
        for text, images in zip(texts, images_list):
            num_images = len(images)
            placeholders = "\n".join("(<image>./</image>)" for _ in range(num_images))
            if placeholders and text:
                content = f"{placeholders}\n{text}"
            elif placeholders:
                content = placeholders
            else:
                content = text

            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        return ModelRequestData(prompts=prompts, stop_token_ids=stop_token_ids)


# ── Test helpers ──────────────────────────────────────────────────


def _generate_dummy_images(count: int) -> list[Image.Image]:
    """Return placeholder PIL images for prompt-construction tests."""
    return [Image.new("RGB", (1, 1), color=0) for _ in range(count)]


def _parse_cli_args() -> argparse.Namespace:
    all_models = [
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "moonshotai/Kimi-VL-A3B-Instruct",
        "deepseek-ai/deepseek-vl2",
        "zai-org/GLM-4.5V",
        "openbmb/MiniCPM-o-2_6",
        "microsoft/Phi-4-multimodal-instruct",
        "MIL-UT/Asagi-14B",
        *LLAVA_1_5_MODELS,
        *LLAVA_NEXT_MODELS,
        *PANGEA_MODELS,
        *QWEN2_VL_MODELS,
        *QWEN2_5_VL_MODELS,
        *INTERNVL2_MODELS,
        *INTERNVL3_MODELS,
        *GEMMA3_MODELS,
        *AYA_VISION_MODELS,
        *OVIS2_MODELS,
        *OVIS2_5_MODELS,
    ]

    parser = argparse.ArgumentParser(
        description="Preview prompts generated by the VLLM model registry.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        choices=all_models,
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

    images_list = [_generate_dummy_images(count) for count in image_counts]
    registry = VLLMModelRegistry(args.model_id)
    request_data = registry.build_requests(texts, images_list)

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
