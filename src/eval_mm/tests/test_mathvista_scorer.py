"""Tests for MathVistaScorer, extracted from metrics/mathvista_scorer.py."""

from eval_mm.metrics.mathvista_scorer import MathVistaScorer
from eval_mm.metrics.scorer import ScorerConfig


def test_mathvista_scorer():
    scorer = MathVistaScorer(ScorerConfig())

    # Test answer extraction

    # Multiple choice
    response = "Looking at the graph, I can see that the value is 5. The answer is B."
    extracted = scorer.extract_answer(response, 'multi_choice', ['3', '5', '7', '9'])
    assert extracted == 'B', f"Expected 'B', got '{extracted}'"

    # Numeric
    response = "After calculating, the answer is 42.5"
    extracted = scorer.extract_answer(response, 'free_form')
    assert extracted == '42.5', f"Expected '42.5', got '{extracted}'"

    # Test scoring

    # Multiple choice correct
    refs = ["5"]
    preds = ["The answer is B"]
    metadata = [{"question_type": "multi_choice", "answer_type": "text", "choices": ["3", "5", "7", "9"]}]
    scores = scorer.score(refs, preds, metadata)
    assert scores == [1], f"Expected [1], got {scores}"

    # Numeric with precision
    refs = ["3.14159"]
    preds = ["The value is approximately 3.142"]
    metadata = [{"question_type": "free_form", "answer_type": "float", "precision": 2.0}]
    scores = scorer.score(refs, preds, metadata)
    assert scores == [1], f"Expected [1], got {scores}"

    # Integer
    refs = ["1000"]
    preds = ["There are 1,000 items in total"]
    metadata = [{"question_type": "free_form", "answer_type": "integer"}]
    scores = scorer.score(refs, preds, metadata)
    assert scores == [1], f"Expected [1], got {scores}"
