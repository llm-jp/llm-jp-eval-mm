"""Tests for rougel scorer functions, extracted from metrics/rougel_scorer.py."""

import pytest

from eval_mm.metrics.rougel_scorer import RougeLScorer, rouge_ja
from eval_mm.metrics.scorer import ScorerConfig


def test_rouge_ja():
    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    scores = rouge_ja(refs, preds)
    assert scores["rougeL"] == 100.0
    refs = ["たかしが公園で遊んでいた。"]
    preds = ["たかしが公園にいたようだ。"]
    scores = rouge_ja(refs, preds)
    assert pytest.approx(scores["rougeL"], 0.01) == 66.66

    refs = ["私は猫です。", "私は犬です。"]
    preds = ["私は犬です。", "私は猫です。"]
    scores = rouge_ja(refs, preds)
    assert pytest.approx(scores["rougeL"], 0.01) == 80.0
    refs = ["池のほとりです。"]
    preds = ["ここは湖の岸です。"]
    scores = rouge_ja(refs, preds)
    assert pytest.approx(scores["rougeL"], 0.01) == 50.0


def test_rougel_scorer():
    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    scorer = RougeLScorer(ScorerConfig())
    scores = scorer.score(refs, preds)
    assert scores == [100.0]
    refs = ["たかしが公園で遊んでいた。"]
    preds = ["たかしが公園にいたようだ。"]
    scores = scorer.score(refs, preds)
    assert pytest.approx(scores[0], 0.01) == 66.66
    output = scorer.aggregate(scores)
    assert pytest.approx(output.overall_score, 0.01) == 66.66
