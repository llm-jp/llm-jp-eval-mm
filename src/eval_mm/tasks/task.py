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

        # Decide dataset builder at initialization time to avoid runtime
        # introspection in hot paths. Tasks may override `_prepare_test_dataset`
        # for lightweight test-time loading; by default it falls back to
        # `_prepare_dataset`.
        builder = self._prepare_test_dataset if self.is_test_context() else self._prepare_dataset
        ds = builder()
        self.dataset = ds

    def is_test_context(self) -> bool:
        return bool(os.getenv("PYTEST_CURRENT_TEST") or os.getenv("EVAL_MM_TEST_SUBSET") == "1")

    @abc.abstractmethod
    def _prepare_test_dataset(self) -> Dataset:
        """Prepares the test/CI dataset.

        Implementations should load a lightweight subset suitable for
        tests/CI (e.g., slice to first N or stream to collect N examples).
        """
        pass

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
