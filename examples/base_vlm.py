import requests
from PIL import Image
from utils import GenerationConfig


class BaseVLM:
    def __init__(self):
        raise NotImplementedError

    def generate(
        self,
        image: list[Image.Image] | Image.Image,
        prompt: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ):
        """Generate a response given an image (or list of images) and a prompt."""
        raise NotImplementedError

    def test_vlm(self):
        """Test the model with one or two images."""
        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(image_file, stream=True).raw)
        image_file2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        image2 = Image.open(requests.get(image_file2, stream=True).raw)
        print(self.generate(image, "画像には何が映っていますか?"))
        print(self.generate([image, image2], "これらの画像の違いはなんですか?"))
