from abc import ABC, abstractmethod

from PIL import Image
from .generation_config import GenerationConfig


class BaseVLM(ABC):
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def generate(
        self,
        images: list[Image.Image] | None,
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        """Generate a response given an image (or list of images) and a prompt."""
        ...

    @abstractmethod
    def batch_generate(
        self,
        images_list: list[list[Image.Image]] | None,
        text_list: list[str],
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> list[str]:
        """Generate a response given a list of images and a list of prompts."""
        ...
