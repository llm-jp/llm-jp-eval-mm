"""Tests for ExactMatchScorer, extracted from metrics/exact_match_scorer.py."""

from eval_mm.metrics.exact_match_scorer import ExactMatchScorer
from eval_mm.metrics.scorer import ScorerConfig


def test_exact_match_scorer():
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
