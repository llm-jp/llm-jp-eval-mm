from .scorer import Scorer, AggregateOutput
from .scorer_registry import register_scorer


@register_scorer("exact-match")
class ExactMatchScorer(Scorer):
    @staticmethod
    def score(refs: list[str], preds: list[str]) -> list[int]:
        scores = [int(ref == pred) for ref, pred in zip(refs, preds)]
        return scores

    @staticmethod
    def aggregate(scores: list[int]) -> AggregateOutput:
        mean = sum(scores) / len(scores)
        return AggregateOutput(mean, {"exact_match": mean})
