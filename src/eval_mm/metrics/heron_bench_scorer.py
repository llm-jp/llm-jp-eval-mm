from eval_mm.utils.azure_client import OpenAIChatAPI
from collections import defaultdict
import numpy as np
from eval_mm.metrics.scorer import Scorer, AggregateOutput
import re
import json


def parse_score(llm_output: str) -> dict[str, int]:
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            if (
                isinstance(parsed_json, dict)
                and "score" in parsed_json
                and "score_gpt" in parsed_json
            ):
                return {
                    "score": int(parsed_json["score"]),
                    "score_gpt": int(parsed_json["score_gpt"]),
                }
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                if (
                    isinstance(parsed_json, dict)
                    and "score" in parsed_json
                    and "score_gpt" in parsed_json
                ):
                    return {
                        "score": int(parsed_json["score"]),
                        "score_gpt": int(parsed_json["score_gpt"]),
                    }
            except json.JSONDecodeError:
                continue

    return {"score": -1, "score_gpt": -1}


INSTRUCTION = """
You are an expert evaluator. You are given the following information:
- Context: A description of the image.
- Question: A question about the image.
- GPT-4o Answer: GPT-4o's answer to the question.
- Model Answer: The target model's answer to the question.

Your task is to evaluate each answer independently based on how well it answers the question given the context.

Please assign a score from 1 to 10 for each answer according to the following guideline:
- 10: Perfect — Completely correct, relevant, and fully addresses the question based on the context.
- 8-9: Very Good — Mostly correct with only minor inaccuracies or slight omissions.
- 6-7: Good — Generally correct but contains noticeable errors or lacks important details.
- 4-5: Poor — Significant errors or missing key points, but some relevance remains.
- 1-3: Very Poor — Mostly or completely incorrect, irrelevant, or nonsensical.

Output Format (JSON):
Return the result in the following JSON format:
```json
{{
    "score_gpt": int,
    "score": int
}}
```
Do not output anything other than the JSON.

Input:
{{
    "context": {context},
    "question": {question},
    "gpt4o_answer": {gpt4o_answer},
    "model_answer": {model_answer}
}}

Output:
"""


def ask_gpt4_batch(
    content_list: list[str],
    max_tokens: int,
    async_client: OpenAIChatAPI,
    model_name: str,
) -> list[str]:
    message_list = [
        [
            {
                "role": "system",
                "content": "You are a helpful and precise assistant for checking the quality of the answer.",
            },
            {"role": "user", "content": content},
        ]
        for content in content_list
    ]
    completions = async_client.batch_generate_chat_response(
        message_list,
        max_tokens=max_tokens,
        temperature=0,
        seed=0,
        model_name=model_name,
    )
    return completions


class HeronBenchScorer(Scorer):
    def score(self, refs, preds: list[str]) -> list[dict[str, int]]:
        docs = self.config.docs
        assert docs is not None
        assert self.config.client is not None
        assert self.config.judge_model is not None

        contents = [
            INSTRUCTION.format(
                context=doc["context"],
                question=doc["input_text"],
                gpt4o_answer=ref,
                model_answer=pred,
            )
            for doc, ref, pred in zip(docs, refs, preds)
        ]

        completions: list[str] = ask_gpt4_batch(
            contents, 1024, self.config.client, self.config.judge_model
        )

        scores: list[dict[str, int]] = [parse_score(c) for c in completions]
        return scores

    def aggregate(self, scores: list[dict[str, int]]) -> AggregateOutput:
        docs = self.config.docs
        assert docs is not None
        category_list = ["conv", "detail", "complex"]
        heron_metrics = defaultdict(float)
        for category in category_list:
            score_owns = [
                score["score"]
                for score, doc in zip(scores, docs)
                if doc["category"] == category
            ]
            score_gpts = [
                score["score_gpt"]
                for score, doc in zip(scores, docs)
                if doc["category"] == category
            ]
            if len(score_owns) == 0 or np.mean(score_owns) == -1:
                continue
            avg_score = np.mean(score_owns)
            avs_score_rel = (
                100
                * np.mean(score_owns)
                / max(
                    0.01, np.mean(score_gpts)
                )  # divide by 0.01 when 0 division happens
            )
            heron_metrics[category] = avg_score
            heron_metrics[category + "_rel"] = avs_score_rel
        heron_metrics["parse_error_count"] = sum(
            score["score"] == -1 for score in scores
        )
        heron_metrics["overall"] = sum([score["score"] for score in scores]) / len(
            scores
        )
        heron_metrics["overall_rel"] = sum(
            [heron_metrics[category + "_rel"] for category in category_list]
        ) / len(category_list)
        output = AggregateOutput(
            overall_score=heron_metrics["overall_rel"],
            details=heron_metrics,
        )
        return output


def test_heron_bench_scorer():
    from eval_mm.utils.azure_client import MockChatAPI

    refs = ["私は猫です。"]
    preds = ["私は犬です。"]
    docs = [{"context": "hoge", "input_text": "fuga", "category": "conv"}]
    from .scorer import ScorerConfig

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
