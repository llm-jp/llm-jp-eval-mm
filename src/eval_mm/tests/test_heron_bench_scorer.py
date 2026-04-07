"""Tests for HeronBenchScorer, extracted from metrics/heron_bench_scorer.py."""

from eval_mm.metrics.heron_bench_scorer import HeronBenchScorer
from eval_mm.metrics.scorer import ScorerConfig
from eval_mm.utils.azure_client import MockChatAPI


def test_heron_bench_scorer():
    refs = ["私は猫です。"]
    preds = ["私は犬です。"]
    docs = [{"context": "hoge", "input_text": "fuga", "category": "conv"}]
    scorer = HeronBenchScorer(
        ScorerConfig(docs=docs, judge_model="gpt-4o-2024-05-13", client=MockChatAPI())
    )
    scores = scorer.score(refs, preds)
    assert scores == [{"score": -1, "score_gpt": -1}]
    output = scorer.aggregate(scores)
    assert output.overall_score == 0.0
    assert output.details == {
        "parse_error_count": 1,
        "overall": -1.0,
        "conv_rel": 0.0,
        "detail_rel": 0.0,
        "complex_rel": 0.0,
        "overall_rel": 0.0,
    }
