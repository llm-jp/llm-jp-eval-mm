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

        ds = self.prepare_dataset()
        # Apply length cap if requested and if dataset supports len/select
        if self.config.max_dataset_len is not None:
            try:
                n = min(self.config.max_dataset_len, len(ds))  # type: ignore[arg-type]
                ds = ds.select(range(n))  # type: ignore[attr-defined]
            except Exception:
                # Iterable or unknown length; leave as-is (tests should build a small dataset)
                pass
        self.dataset = ds

    def is_test_context(self) -> bool:
        return bool(os.getenv("PYTEST_CURRENT_TEST") or os.getenv("EVAL_MM_TEST_SUBSET") == "1")

    def prepare_dataset(self) -> Dataset:
        """Selects the appropriate dataset builder for the current context.

        If running under pytest/CI, use an optional lightweight
        `_prepare_test_dataset` if the task defines it; otherwise fall back to
        the canonical `_prepare_dataset` implementation.
        """
        if self.is_test_context() and hasattr(self, "_prepare_test_dataset"):
            return getattr(self, "_prepare_test_dataset")()  # type: ignore[misc]
        return self._prepare_dataset()

    def _maybe_slice_split(self, split: str) -> str:
        """Optionally slice HF split to reduce download during tests.

        If running under pytest (PYTEST_CURRENT_TEST present) and a
        max_dataset_len is set in the config, convert e.g. "validation"
        to "validation[:N]" to avoid downloading the full split.
        """
        n = self.config.max_dataset_len
        if n is None:
            return split
        if self.is_test_context():
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
