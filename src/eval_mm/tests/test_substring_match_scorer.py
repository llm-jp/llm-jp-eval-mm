"""Tests for SubstringMatchScorer, extracted from metrics/substring_match_scorer.py."""

from eval_mm.metrics.substring_match_scorer import SubstringMatchScorer
from eval_mm.metrics.scorer import ScorerConfig


def test_substring_match_scorer():
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

    # Test with list of valid answers (like DocVQA)
    refs = [["university of california", "UC"], ["0.28", "0.280"]]
    preds = ["The university of california is great", "The value is 0.280"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1]  # Both should match

    refs = [["apple", "orange"], ["cat", "dog"]]
    preds = ["I like bananas", "I have a cat"]
    scores = scorer.score(refs, preds)
    assert scores == [0, 1]  # First no match, second matches "cat"

    # Test case-insensitive matching
    refs = ["Hello", ["World", "EARTH"], "Python"]
    preds = ["hello there", "This is the world", "I love PYTHON"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1]  # All should match with case-insensitive

    refs = ["Tokyo", ["New York", "NYC"]]
    preds = ["TOKYO is big", "I visited new york"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1]  # Both should match with case-insensitive

    # Test number word conversion
    refs = ["2", "5", ["10", "ten"], "20"]
    preds = ["There are two apples", "I have five dollars", "ten items", "twenty people"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1]  # All should match with number conversion

    # Test mixed number formats
    refs = ["three", ["4", "four"], "100"]
    preds = ["I see 3 cats", "There are four birds", "one hundred percent"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1]  # All should match

    # Test that partial matches are avoided
    refs = ["one"]
    preds = ["someone is here"]  # "one" in "someone" should not match
    scores = scorer.score(refs, preds)
    assert scores == [0]  # Should not match due to word boundaries
