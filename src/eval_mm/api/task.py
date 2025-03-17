import abc

from dataclasses import dataclass
from eval_mm.utils.azure_client import OpenAIChatAPI
from datasets import Dataset


@dataclass
class TaskConfig:
    max_dataset_len: int | None = None
    judge_model: str = "gpt-4o-mini-2024-07-18"
    batch_size_for_evaluation: int = 10
    rotate_choices: bool = False


class Task(abc.ABC):
    """A task represents an entire task including its dataset, problems, answers, and evaluation methods.
    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
    {"question": ..., "answer": ...} or {"question": ..., question, answer)
    """

    def __init__(self, config: TaskConfig):
        self._dataset = None
        self.client = OpenAIChatAPI()
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
    def doc_to_text(self, doc):
        """Converts a document to text."""
        pass

    @abc.abstractmethod
    def doc_to_visual(self, doc):
        """Converts a document to visual."""
        pass

    @abc.abstractmethod
    def doc_to_id(self, doc):
        """Converts a document to id."""
        pass

    @abc.abstractmethod
    def doc_to_answer(self, doc):
        """Converts a document to answer."""
        pass

    @abc.abstractmethod
    def calc_scores(self, preds: list, metric: str) -> list:
        """Calculates scores for the predictions."""
        pass

    @abc.abstractmethod
    def gather_scores(self, scores: list[dict], metric: str) -> dict:
        """Aggregates the scores."""
        pass
