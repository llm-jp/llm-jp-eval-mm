"""Tests for AI2DScorer, extracted from metrics/ai2d_scorer.py."""

from eval_mm.metrics.ai2d_scorer import AI2DScorer
from eval_mm.metrics.scorer import ScorerConfig


def test_ai2d_scorer():
    scorer = AI2DScorer(ScorerConfig())

    # Test basic functionality
    refs = ["A", "B", "C", "D"]
    preds = ["A", "B", "C", "D"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1]

    # Test normalization - spaces
    refs = ["A", "B", "C", "D"]
    preds = [" A ", "B ", " C", "D"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1]

    # Test normalization - periods
    refs = ["A", "B", "C", "D"]
    preds = ["A.", "B.", "C.", "D."]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1]

    # Test normalization - lowercase
    refs = ["A", "B", "C", "D"]
    preds = ["a", "b", "c", "d"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1]

    # Test mixed normalization
    refs = ["A", "B", "C", "D"]
    preds = [" a. ", "b.", " C ", "d"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1]

    # Test incorrect answers
    refs = ["A", "B", "C", "D"]
    preds = ["B", "A", "D", "C"]
    scores = scorer.score(refs, preds)
    assert scores == [0, 0, 0, 0]

    # Test invalid answers
    refs = ["A", "B", "C", "D"]
    preds = ["E", "1", "X", ""]
    scores = scorer.score(refs, preds)
    assert scores == [0, 0, 0, 0]

    # Test aggregation
    output = scorer.aggregate([1, 1, 1, 0])
    assert output.overall_score == 0.75
    assert output.details == {"accuracy": 0.75}

    output = scorer.aggregate([1, 0, 0, 0])
    assert output.overall_score == 0.25
    assert output.details == {"accuracy": 0.25}

    # Test empty scores
    output = scorer.aggregate([])
    assert output.overall_score == 0.0
    assert output.details == {"accuracy": 0.0}
