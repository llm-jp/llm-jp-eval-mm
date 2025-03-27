from .scorer import Scorer, AggregateOutput


class SubstringMatchScorer(Scorer):
    @staticmethod
    def score(refs: list[str], preds: list[str]) -> list[int]:
        scores = [int(ref in pred) for ref, pred in zip(refs, preds)]
        return scores

    @staticmethod
    def aggregate(scores: list[int]) -> AggregateOutput:
        mean = sum(scores) / len(scores)
        return AggregateOutput(mean, {"substring_match": mean})


def test_substring_match_scorer():
    from .scorer import ScorerConfig

    scorer = SubstringMatchScorer(ScorerConfig())
    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    scores = scorer.score(refs, preds)
    assert scores == [1]
    refs = ["たかしが公園で遊んでいた。"]
    preds = ["たかしが公園にいたようだ。"]
    scores = scorer.score(refs, preds)
    assert scores == [0]
    refs = ["私は猫です。", "私は犬です。"]
    preds = ["私は犬です。", "私は猫です。"]
    scores = scorer.score(refs, preds)
    assert scores == [0, 0]
    refs = ["池のほとりです。"]
    preds = ["ここは湖の岸です。"]
    scores = scorer.score(refs, preds)
    assert scores == [0]

    output = scorer.aggregate([1, 1, 1, 0])
    assert output.overall_score == 0.75
    assert output.details == {"substring_match": 0.75}
