from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from io import BytesIO
import re
from typing import Literal

from PIL import Image
from PIL import ImageDraw
from utils import GenerationConfig
from loguru import logger

SmokeTestMode = Literal["offline", "online"]


@dataclass(frozen=True)
class SmokeTestCase:
    images: list[Image.Image]
    prompt: str


class BaseVLM:
    _LEADING_THINK_BLOCK_RE = re.compile(
        r"^\s*(?:<think\b[^>]*>.*?</think>\s*)+",
        flags=re.DOTALL | re.IGNORECASE,
    )

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._wrap_generation_method("generate")
        cls._wrap_generation_method("batch_generate")

    def __init__(self):
        raise NotImplementedError

    def generate(
        self,
        images: list[Image.Image] | None,
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        """Generate a response given an image (or list of images) and a prompt."""
        raise NotImplementedError

    def batch_generate(
        self,
        images_list: list[list[Image.Image]] | None,
        text_list: list[str],
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> list[str]:
        """Generate a response given a list of images and a list of prompts."""
        raise NotImplementedError

    @classmethod
    def _wrap_generation_method(cls, method_name: str) -> None:
        method = cls.__dict__.get(method_name)
        if method is None or getattr(method, "_base_vlm_wrapped", False):
            return

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            if isinstance(result, str):
                return self.postprocess_generation(result)
            if isinstance(result, list):
                return [
                    self.postprocess_generation(item) if isinstance(item, str) else item
                    for item in result
                ]
            return result

        wrapper._base_vlm_wrapped = True
        setattr(cls, method_name, wrapper)

    def postprocess_generation(self, text: str) -> str:
        """Normalize model output before it is saved or scored."""
        return self._LEADING_THINK_BLOCK_RE.sub("", text).strip()

    def get_smoke_test_cases(
        self, smoke_test_mode: SmokeTestMode = "offline"
    ) -> list[SmokeTestCase]:
        if smoke_test_mode == "online":
            return self._build_online_smoke_test_cases()
        return self._build_offline_smoke_test_cases()

    def _build_offline_smoke_test_cases(self) -> list[SmokeTestCase]:
        image_a = self._make_test_image(
            background="#F6D365",
            accent="#EA5F89",
            label="A",
        )
        image_b = self._make_test_image(
            background="#5B8DEF",
            accent="#5AD8A6",
            label="B",
        )
        return [
            SmokeTestCase(
                images=[image_a],
                prompt="画像を一文で簡潔に説明してください。",
            ),
            SmokeTestCase(
                images=[image_a, image_b],
                prompt="2枚の画像の違いを一文で説明してください。",
            ),
        ]

    def _build_online_smoke_test_cases(self) -> list[SmokeTestCase]:
        image = self._load_image_from_url(
            "http://images.cocodataset.org/val2017/000000039769.jpg"
        )
        image2 = self._load_image_from_url(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        )
        return [
            SmokeTestCase(
                images=[image],
                prompt="画像には何が映っていますか?",
            ),
            SmokeTestCase(
                images=[image, image2],
                prompt="これらの画像の違いはなんですか?",
            ),
        ]

    def _make_test_image(
        self,
        background: str,
        accent: str,
        label: str,
        size: tuple[int, int] = (256, 256),
    ) -> Image.Image:
        image = Image.new("RGB", size, background)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((24, 24, 232, 232), radius=28, outline=accent, width=8)
        draw.ellipse((56, 72, 136, 152), fill=accent)
        draw.rectangle((152, 88, 208, 184), fill=accent)
        draw.text((110, 184), label, fill="black")
        return image

    def _load_image_from_url(self, url: str) -> Image.Image:
        import requests

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    def test_vlm(self, smoke_test_mode: SmokeTestMode = "offline") -> None:
        """Test the model with local fixtures by default."""
        for case in self.get_smoke_test_cases(smoke_test_mode):
            output = self.generate(case.images, case.prompt)
            logger.info(f"Output: {output}")
            assert isinstance(output, str), (
                f"Expected output to be a string, but got {type(output)}"
            )

    def test_vlm_100(self, smoke_test_mode: SmokeTestMode = "offline") -> None:
        """Run the single-image smoke case repeatedly."""
        case = self.get_smoke_test_cases(smoke_test_mode)[0]
        import time

        start_time = time.time()
        for _ in range(100):
            output = self.generate(case.images, case.prompt)
            logger.info(f"Output: {output}")
            assert isinstance(output, str), (
                f"Expected output to be a string, but got {type(output)}"
            )
        end_time = time.time()
        logger.info(f"Time taken: {end_time - start_time} seconds for 100 times")

    def test_vlm_batch_100(self, smoke_test_mode: SmokeTestMode = "offline") -> None:
        """Run the single-image smoke case in batch mode."""
        print("=== Batch 100 test ===")
        print(f"Model: {self.model_id}")
        case = self.get_smoke_test_cases(smoke_test_mode)[0]

        import time

        image_list = [list(case.images) for _ in range(100)]
        text_list = [case.prompt for _ in range(100)]

        start_time = time.time()
        outputs = self.batch_generate(image_list, text_list)
        for output in outputs:
            assert isinstance(output, str), (
                f"Expected output to be a string, but got {type(output)}"
            )

        end_time = time.time()
        logger.info(f"Time taken: {end_time - start_time} seconds for BATCH 100 times")


class _DummyVLM(BaseVLM):
    def __init__(self) -> None:
        self.model_id = "dummy"

    def generate(
        self,
        images: list[Image.Image] | None,
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        return "<think>hidden reasoning</think>\n  answer  "

    def batch_generate(
        self,
        images_list: list[list[Image.Image]] | None,
        text_list: list[str],
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> list[str]:
        return [
            "<think>hidden reasoning</think>\nfirst",
            " second ",
        ]


def test_generate_postprocesses_leading_think_blocks() -> None:
    model = _DummyVLM()
    assert model.generate([], "prompt") == "answer"


def test_batch_generate_postprocesses_each_item() -> None:
    model = _DummyVLM()
    assert model.batch_generate([[], []], ["prompt1", "prompt2"]) == ["first", "second"]


def test_offline_smoke_cases_are_local_rgb_images() -> None:
    model = _DummyVLM()
    cases = model.get_smoke_test_cases("offline")
    assert len(cases) == 2
    assert [len(case.images) for case in cases] == [1, 2]
    assert all(image.mode == "RGB" for case in cases for image in case.images)
