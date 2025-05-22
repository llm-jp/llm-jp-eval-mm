import requests
from PIL import Image
from utils import GenerationConfig
from loguru import logger


class BaseVLM:
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

    def test_vlm(self):
        """Test the model with one or two images."""
        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(image_file, stream=True).raw)
        image_file2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        image2 = Image.open(requests.get(image_file2, stream=True).raw)
        output = self.generate([image], "画像には何が映っていますか?")
        logger.info(f"Output: {output}")
        assert isinstance(
            output, str
        ), f"Expected output to be a string, but got {type(output)}"

        output = self.generate([image, image2], "これらの画像の違いはなんですか?")
        logger.info(f"Output: {output}")
        assert isinstance(
            output, str
        ), f"Expected output to be a string, but got {type(output)}"

        # --- No image case ---
        # output = self.generate([], "画像には何が映っていますか?")
        # logger.info(f"Output: {output}")
        # assert isinstance(
        #     output, str
        # ), f"Expected output to be a string, but got {type(output)}"

    def test_vlm_100(self):
        """Test the model with one or two images."""
        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(image_file, stream=True).raw)

        import time

        start_time = time.time()
        for _ in range(100):
            output = self.generate([image], "画像には何が映っていますか?")
            logger.info(f"Output: {output}")
            assert isinstance(
                output, str
            ), f"Expected output to be a string, but got {type(output)}"
        end_time = time.time()
        logger.info(f"Time taken: {end_time - start_time} seconds for 100 times")

    def test_vlm_batch_100(self):
        """Test the model with one or two images."""

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
            assert isinstance(
                output, str
            ), f"Expected output to be a string, but got {type(output)}"

        end_time = time.time()
        logger.info(f"Time taken: {end_time - start_time} seconds for BATCH 100 times")
