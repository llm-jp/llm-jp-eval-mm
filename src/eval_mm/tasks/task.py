import abc

from dataclasses import dataclass
from datasets import Dataset
from PIL import Image


@dataclass
class TaskConfig:
    max_dataset_len: int | None = None
    rotate_choices: bool = False


class Task(abc.ABC):
    def __init__(self, config: TaskConfig):
        self.config = config

        if self.config.max_dataset_len is not None:
            self.dataset = self._prepare_dataset().select(
                range(self.config.max_dataset_len)
            )
        else:
            self.dataset = self._prepare_dataset()

    @abc.abstractmethod
    def _prepare_dataset(self) -> Dataset:
        """Prepares the dataset."""
        pass

    @abc.abstractmethod
    def doc_to_text(self, doc) -> str:
        """Converts a document to text."""
        pass

    @abc.abstractmethod
    def doc_to_visual(self, doc) -> list[Image.Image]:
        """Converts a document to visual."""
        pass

    @abc.abstractmethod
    def doc_to_id(self, doc) -> str:
        """Converts a document to id."""
        pass

    @abc.abstractmethod
    def doc_to_answer(self, doc) -> str:
        """Converts a document to answer."""
        pass
