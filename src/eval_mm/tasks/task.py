import abc
import os

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

    def _maybe_slice_split(self, split: str) -> str:
        """Optionally slice HF split to reduce download during tests.

        If running under pytest (PYTEST_CURRENT_TEST present) and a
        max_dataset_len is set in the config, convert e.g. "validation"
        to "validation[:N]" to avoid downloading the full split.
        """
        n = self.config.max_dataset_len
        if n is None:
            return split
        if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("EVAL_MM_TEST_SUBSET") == "1":
            # Respect existing slice if provided
            if "[" in split:
                return split
            return f"{split}[:{n}]"
        return split

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
