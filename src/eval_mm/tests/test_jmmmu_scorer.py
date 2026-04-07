"""Tests for JMMMUScorer, extracted from metrics/jmmmu_scorer.py."""

from eval_mm.metrics.jmmmu_scorer import JMMMUScorer
from eval_mm.metrics.scorer import ScorerConfig


def test_jmmmu_scorer():
    refs = ["A"]
    preds = ["A"]
    docs = [
        {
            "question_type": "multiple-choice",
            "options": '["A", "B", "C", "D"]',
            "answer": "A",
            "id": "validation_Accounting_1",
        }
    ]
    scorer = JMMMUScorer(ScorerConfig(docs=docs))
    scores = scorer.score(refs, preds)
    assert scores == [1]
    output = scorer.aggregate(scores)
    assert output.overall_score == 1.0
    assert output.details == {
        "Overall-Art and Psychology": 0,
        "Overall-Business": 1.0,
        "Accounting": 1.0,
        "Overall-Science": 0,
        "Overall-Health and Medicine": 0,
        "Overall-Tech and Engineering": 0,
        "Overall": 1.0,
    }
