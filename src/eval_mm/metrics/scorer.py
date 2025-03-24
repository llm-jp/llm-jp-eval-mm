from abc import ABC, abstractmethod
from dataclasses import dataclass
from eval_mm.utils.azure_client import OpenAIChatAPI


@dataclass
class AggregateOutput:
    overall_score: float
    details: dict[str, float]


@dataclass
class ScorerConfig:
    docs: dict = None
    judge_model: str = None
    client: OpenAIChatAPI = None
    batch_size: int = 10


class Scorer(ABC):
    def __init__(self, config: ScorerConfig):
        self.config = config

    @abstractmethod
    def score(self, refs: list[str], preds: list[str]) -> list:
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, scores: list) -> AggregateOutput:
        raise NotImplementedError
