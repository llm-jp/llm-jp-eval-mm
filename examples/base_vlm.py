"""Backward-compatible re-export. Use ``from eval_mm import BaseVLM`` instead."""

import requests
from PIL import Image
from loguru import logger

from eval_mm.models.base_vlm import BaseVLM as _BaseVLM
from eval_mm.models.generation_config import GenerationConfig


class BaseVLM(_BaseVLM):
    """BaseVLM with convenience test helpers for examples/ adapters."""

    def batch_generate(
        self,
        images_list: list[list[Image.Image]] | None,
        text_list: list[str],
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> list[str]:
        """Default batch_generate: sequential fallback for adapters that only implement generate."""
        if images_list is None:
            images_list = [[] for _ in text_list]
        return [
            self.generate(images, text, gen_kwargs)
            for images, text in zip(images_list, text_list)
        ]

    def test_vlm(self):
        """Test the model with one or two images."""
        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(image_file, stream=True).raw)
        image_file2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        image2 = Image.open(requests.get(image_file2, stream=True).raw)
        output = self.generate([image], "画像には何が映っていますか?")
        logger.info(f"Output: {output}")
        assert isinstance(output, str), f"Expected str, got {type(output)}"

        output = self.generate([image, image2], "これらの画像の違いはなんですか?")
        logger.info(f"Output: {output}")
        assert isinstance(output, str), f"Expected str, got {type(output)}"

    def test_vlm_100(self):
        """Test the model 100 times sequentially."""
        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(image_file, stream=True).raw)

        import time

        start_time = time.time()
        for _ in range(100):
            output = self.generate([image], "画像には何が映っていますか?")
            logger.info(f"Output: {output}")
            assert isinstance(output, str), f"Expected str, got {type(output)}"
        end_time = time.time()
        logger.info(f"Time taken: {end_time - start_time} seconds for 100 times")

    def test_vlm_batch_100(self):
        """Test the model with 100 batch items."""
        print("=== Batch 100 test ===")
        print(f"Model: {self.model_id}")

        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(image_file, stream=True).raw)

        import time

        image_list = [[image] for _ in range(100)]
        text_list = [["画像には何が映っていますか?"] for _ in range(100)]

        start_time = time.time()
        outputs = self.batch_generate(image_list, text_list)
        for output in outputs:
            assert isinstance(output, str), f"Expected str, got {type(output)}"
        end_time = time.time()
        logger.info(f"Time taken: {end_time - start_time} seconds for BATCH 100 times")
