"""Tests for JICVQAScorer, extracted from metrics/jic_vqa_scorer.py."""

from eval_mm.metrics.jic_vqa_scorer import JICVQAScorer
from eval_mm.metrics.scorer import ScorerConfig


def test_jic_vqa_scorer():
    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    scorer = JICVQAScorer(ScorerConfig(docs={"domain": ["test"]}))
    scores = scorer.score(refs, preds)
    assert scores == [1]
    output = scorer.aggregate(scores)
    assert output.overall_score == 1.0
    assert output.details == {"test": 1.0, "average": 1.0}
