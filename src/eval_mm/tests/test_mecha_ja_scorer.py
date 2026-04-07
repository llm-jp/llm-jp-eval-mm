"""Tests for MECHAJaScorer, extracted from metrics/mecha_ja_scorer.py."""

from eval_mm.metrics.mecha_ja_scorer import MECHAJaScorer
from eval_mm.metrics.scorer import ScorerConfig


def test_mechaja_scorer():
    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    scorer = MECHAJaScorer(
        ScorerConfig(
            docs=[{"question_id": "q1", "answer_type": 0, "background_text": ""}]
        )
    )
    scores = MECHAJaScorer.score(refs, preds)

    assert scores == [1]
    output = scorer.aggregate(scores)
    assert output.overall_score == 1.0
    assert output.details == {
        "overall": 1.0,
        "Factoid": 1.0,
        "Non-Factoid": 0.0,
        "with_background": 0.0,
        "without_background": 1.0,
        "rot_no_rot": {
            "overall": 1.0,
            "Factoid": 1.0,
            "Non-Factoid": 0.0,
            "with_background": 0.0,
            "without_background": 1.0,
        },
    }
