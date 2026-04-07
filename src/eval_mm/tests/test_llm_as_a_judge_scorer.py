"""Tests for LlmAsaJudgeScorer, extracted from metrics/llm_as_a_judge_scorer.py."""

from eval_mm.metrics.llm_as_a_judge_scorer import LlmAsaJudgeScorer
from eval_mm.metrics.scorer import ScorerConfig
from eval_mm.utils.azure_client import MockChatAPI


def test_llm_as_a_judge_scorer():
    questions = ["What is the capital of Japan?", "What is the capital of France?"]
    answers = ["Tokyo", "Paris"]
    preds = ["", ""]
    scorer = LlmAsaJudgeScorer(
        ScorerConfig(docs={"input_text": questions}, client=MockChatAPI())
    )
    scores = scorer.score(answers, preds)
    assert scores == [0, 0]
    output = scorer.aggregate(scores)
    assert output.overall_score == 0.0
    assert output.details == {"llm_as_a_judge": 0.0}
