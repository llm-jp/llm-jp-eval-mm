"""Tests for JDocQAScorer, extracted from metrics/jdocqa_scorer.py."""

from eval_mm.metrics.jdocqa_scorer import JDocQAScorer
from eval_mm.metrics.scorer import ScorerConfig


def test_jdocqa_scorer():
    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    scorer = JDocQAScorer(ScorerConfig(docs=[{"answer_type": 1}]))
    scores = scorer.score(refs, preds)
    assert scores == [1.0]
    output = scorer.aggregate(scores)
    assert output.overall_score == 1.0
    assert output.details == {
        "factoid_exact": 1.0,
        "yesno_exact": 0,
        "numerical_exact": 0,
        "open-ended_bleu": 0,
        "overall": 1.0,
    }
