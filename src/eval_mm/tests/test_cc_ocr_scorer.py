"""Tests for CCOCRScorer, extracted from metrics/cc_ocr_scorer.py."""

from typing import List

from eval_mm.metrics.cc_ocr_scorer import CCOCRScorer
from eval_mm.metrics.scorer import ScorerConfig


def test_cc_ocr_scorer():
    config = ScorerConfig()
    scorer = CCOCRScorer(config)

    # Test 1: Exact match (Japanese example)
    refs1 = ["これはテストです。", "第二の例です。"]
    preds1 = ["これはテストです。", "第二の例です。"]
    scores1 = scorer.score(refs1, preds1)
    assert all(
        abs(s - 1.0) < 1e-6 for s in scores1
    ), f"Test 1 Scores: Expected all 1.0, got {scores1}"
    agg_output1 = scorer.aggregate(scores1)
    assert abs(agg_output1.overall_score - 1.0) < 1e-6, "Test 1 Overall Score"
    assert abs(agg_output1.details["micro_f1_score"] - 1.0) < 1e-6, "Test 1 Micro F1"
    assert abs(agg_output1.details["macro_f1_score"] - 1.0) < 1e-6, "Test 1 Macro F1"

    # Test 2: Partial match (Japanese example, character-level)
    refs2 = ["リンゴ バナナ オレンジ", "猫 犬"]
    preds2 = ["リンゴ バ ナナ", "猫 犬 鳥"]
    scores2 = scorer.score(refs2, preds2)
    expected_scores2 = [0.75, 0.8]
    for s, es in zip(scores2, expected_scores2):
        assert abs(s - es) < 1e-6, f"Test 2 Sample Score: Expected {es}, got {s}"

    agg_output2 = scorer.aggregate(scores2)
    assert (
        abs(agg_output2.overall_score - 0.775) < 1e-6
    ), f"Test 2 Overall Score: Expected 0.775, Got {agg_output2.overall_score}"

    expected_micro_f1_2 = 16 / 21
    assert (
        abs(agg_output2.details["micro_f1_score"] - expected_micro_f1_2) < 1e-6
    ), f"Test 2 Micro F1: Expected {expected_micro_f1_2}, Got {agg_output2.details['micro_f1_score']}"

    expected_macro_f1_2 = 0.775
    assert (
        abs(agg_output2.details["macro_f1_score"] - expected_macro_f1_2) < 1e-6
    ), f"Test 2 Macro F1: Expected {expected_macro_f1_2}, Got {agg_output2.details['macro_f1_score']}"

    # Test 3: Mismatch and normalization (Japanese)
    refs3 = [
        "こんにちは\u3000世界",
        "吾輩は猫である。名前はまだ無い。",
    ]
    preds3 = [
        "こんにちわ世界",
        "吾輩は猫である。名前はまだ無 い。",
    ]

    f1_s1 = 6 / 7
    f1_s2 = 1.0

    scores3 = scorer.score(refs3, preds3)
    expected_scores3 = [f1_s1, f1_s2]
    for s, es in zip(scores3, expected_scores3):
        assert abs(s - es) < 1e-6, f"Test 3 Sample Score: Expected {es}, got {s}"

    agg_output3 = scorer.aggregate(scores3)
    expected_overall3 = 13 / 14
    assert (
        abs(agg_output3.overall_score - expected_overall3) < 1e-6
    ), f"Test 3 Overall Score: Expected {expected_overall3}, Got {agg_output3.overall_score}"

    expected_micro_f1_3 = 22 / 23
    assert (
        abs(agg_output3.details["micro_f1_score"] - expected_micro_f1_3) < 1e-6
    ), f"Test 3 Micro F1: Expected {expected_micro_f1_3}, Got {agg_output3.details['micro_f1_score']}"

    expected_macro_f1_3 = 13 / 14
    assert (
        abs(agg_output3.details["macro_f1_score"] - expected_macro_f1_3) < 1e-6
    ), f"Test 3 Macro F1: Expected {expected_macro_f1_3}, Got {agg_output3.details['macro_f1_score']}"

    # Test 4: Empty input
    refs_empty: List[str] = []
    preds_empty: List[str] = []
    scores_empty = scorer.score(refs_empty, preds_empty)
    assert scores_empty == [], "Test 4 Scores Empty"
    agg_output_empty = scorer.aggregate(scores_empty)
    assert agg_output_empty.overall_score == 0.0, "Test 4 Overall Score Empty"
    assert agg_output_empty.details["micro_f1_score"] == 0.0, "Test 4 Micro F1 Empty"
    assert agg_output_empty.details["macro_f1_score"] == 0.0, "Test 4 Macro F1 Empty"
