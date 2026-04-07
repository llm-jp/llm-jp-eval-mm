"""Tests for BLINKScorer, extracted from metrics/blink_scorer.py."""

from eval_mm.metrics.blink_scorer import BLINKScorer
from eval_mm.metrics.scorer import ScorerConfig


def test_blink_scorer():
    """Test BLINK scorer functionality."""
    scorer = BLINKScorer(ScorerConfig())

    # Test basic functionality with BLINK format
    refs = ["A", "B", "C", "D"]
    preds = ["(A)", "(B)", "(C)", "(D)"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1], f"Expected all correct, got {scores}"

    # Test normalization - parentheses removal
    refs = ["A", "B", "C", "D"]
    preds = ["(A)", "B", "(C)", "D"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1], f"Expected all correct, got {scores}"

    # Test normalization - spaces
    refs = ["A", "B", "C", "D"]
    preds = [" A ", "B ", " C", "D"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1], f"Expected all correct, got {scores}"

    # Test normalization - periods
    refs = ["A", "B", "C", "D"]
    preds = ["A.", "B.", "C.", "D."]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1], f"Expected all correct, got {scores}"

    # Test normalization - lowercase
    refs = ["A", "B", "C", "D"]
    preds = ["a", "b", "c", "d"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1], f"Expected all correct, got {scores}"

    # Test mixed normalization
    refs = ["A", "B", "C", "D"]
    preds = [" (a) ", "b.", " C ", "(D)."]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1], f"Expected all correct, got {scores}"

    # Test incorrect answers
    refs = ["A", "B", "C", "D"]
    preds = ["B", "A", "D", "C"]
    scores = scorer.score(refs, preds)
    assert scores == [0, 0, 0, 0], f"Expected all incorrect, got {scores}"

    # Test invalid answers
    refs = ["A", "B", "C", "D"]
    preds = ["E", "1", "X", ""]
    scores = scorer.score(refs, preds)
    assert scores == [0, 0, 0, 0], f"Expected all incorrect, got {scores}"

    # Test aggregation
    output = scorer.aggregate([1, 1, 1, 0])
    assert output.overall_score == 0.75, f"Expected 0.75, got {output.overall_score}"
    assert output.details["accuracy"] == 0.75
    assert output.details["total_samples"] == 4
    assert output.details["correct_samples"] == 3

    # Test empty scores
    output = scorer.aggregate([])
    assert output.overall_score == 0.0
    assert output.details["accuracy"] == 0.0
