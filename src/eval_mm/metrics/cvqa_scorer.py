# cvqa_scorer.py
from .scorer import Scorer, AggregateOutput, ScorerConfig


class CVQAScorer(Scorer):
    @staticmethod
    def score(refs: list[str], preds: list[str]) -> list[int]:
        """
        Checks whether each reference string is contained in the corresponding
        prediction string and returns a list of integer scores (1 for True, 0 for False).
        """
        scores = []
        for ref, pred in zip(refs, preds):
            score = 1 if ref in pred else 0
            scores.append(score)
        return scores

    def aggregate(self, scores: list[int]) -> AggregateOutput:
        overall_score = sum(scores) / len(scores)
        details = {"overall": overall_score}
        output = AggregateOutput(overall_score, details)
        return output


def test_cvqa_scorer():
    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    scorer = CVQAScorer(ScorerConfig())
    scores = CVQAScorer.score(refs, preds)
    assert scores == [1]
    output = scorer.aggregate(scores)
    assert output.overall_score == 1.0
    assert output.details == {
        "overall": 1.0,
    }
