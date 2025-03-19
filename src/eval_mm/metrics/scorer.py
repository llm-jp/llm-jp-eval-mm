from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AggregateOutput:
    overall_score: float
    details: dict[str, float]


class Scorer(ABC):
    def __init__(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def score(refs: list[str], preds: list[str], **kwargs) -> list[int | float]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def aggregate(scores: list, **kwargs) -> AggregateOutput:
        raise NotImplementedError
