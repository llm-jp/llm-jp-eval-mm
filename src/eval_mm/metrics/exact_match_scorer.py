from .scorer import Scorer, AggregateOutput


class ExactMatchScorer(Scorer):
    @staticmethod
    def score(refs: list[str], preds: list[str]) -> list[int]:
        scores = [int(ref == pred) for ref, pred in zip(refs, preds)]
        return scores

    @staticmethod
    def aggregate(scores: list[int]) -> AggregateOutput:
        mean = sum(scores) / len(scores)
        return AggregateOutput(mean, {"exact_match": mean})


def test_exact_match_scorer():
    from .scorer import ScorerConfig

    scorer = ExactMatchScorer(ScorerConfig())
    refs = ["私は猫です。", "私は犬です。"]
    preds = ["私は犬です。", "私は犬です。"]
    scores = scorer.score(refs, preds)
    assert scores == [0, 1]
    scores = scorer.aggregate([1, 1, 1, 0])
    assert scores.overall_score == 0.75
    assert scores.details == {"exact_match": 0.75}
    scores = scorer.aggregate([1, 1, 0, 0])
    assert scores.overall_score == 0.5
    assert scores.details == {"exact_match": 0.5}
