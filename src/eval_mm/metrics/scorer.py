from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict
from eval_mm.utils.azure_client import OpenAIChatAPI


@dataclass
class AggregateOutput:
    overall_score: float
    details: dict[str, float]


@dataclass
class ScorerConfig:
    docs: Optional[Dict] = None
    judge_model: Optional[str] = None
    client: Optional[OpenAIChatAPI] = None
    batch_size: int = 10
    random_choice: bool = False


class Scorer(ABC):
    def __init__(self, config: ScorerConfig):
        self.config = config

    @abstractmethod
    def score(self, refs: list[str], preds: list[str]) -> list:
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, scores: list) -> AggregateOutput:
        raise NotImplementedError
