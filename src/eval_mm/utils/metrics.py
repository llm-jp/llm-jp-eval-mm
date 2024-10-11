# Reference: https://github.com/SakanaAI/evolutionary-model-merge/blob/main/evomerge/eval/metrics.py

import re
from rouge_score import rouge_scorer, scoring
from fugashi import Tagger
import emoji
import unicodedata
# import neologdn FIXME: fix c++12 error when installing neologdn


class MecabTokenizer:
    def __init__(self) -> None:
        self.tagger = Tagger("-Owakati")

    def normalize_answer(self, text: str) -> str:
        """Lower case text, remove punctuation and extra whitespace, etc."""

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_emoji(text: str) -> str:
            text = "".join(["" if emoji.is_emoji(c) else c for c in text])
            emoji_pattern = re.compile(
                "["
                "\U0001f600-\U0001f64f"  # emoticons
                "\U0001f300-\U0001f5ff"  # symbols & pictographs
                "\U0001f680-\U0001f6ff"  # transport & map symbols
                "\U0001f1e0-\U0001f1ff"  # flags (iOS)
                "\U00002702-\U000027b0"
                "]+",
                flags=re.UNICODE,
            )
            return emoji_pattern.sub(r"", text)

        text = remove_emoji(text)
        # see neologdn docs for details, but handles things like full/half width variation
        # text = neologdn.normalize(text) FIXME: fix c++12 error when installing neologdn
        text = unicodedata.normalize("NFKC", text)
        text = white_space_fix(text)
        return text

    def tokenize(self, text):
        return self.tagger.parse(self.normalize_answer(text)).split()


def rouge_ja(refs: list[str], preds: list[str]) -> dict:
    """Compute ROUGE-L scores for Japanese text.
    Args:
        refs: list of reference strings
        preds: list of predicted strings
    Returns:
        dict: dictionary with keys: { 'rouge1', 'rouge2', 'rougeL' }
        Each value is a float representing the ROUGE score (f-measure) * 100.
    """
    assert isinstance(refs, list) and isinstance(
        preds, list
    ), "refs and preds must be lists."
    tokenizer = MecabTokenizer()
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    # mecab-based rouge
    scorer = rouge_scorer.RougeScorer(
        rouge_types,
        tokenizer=tokenizer,
    )

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}


def llm_as_a_judge(client, template, questions, answers, preds):
    """Evaluat
    Reference:
    注: 評価方法はGPT-4oによるスコアリング方法を採用しました。各設問ごとに5点満点で評価するようGPT-4oに指示を出し、平均点をモデルのスコアとしています。値が高いほど複数画像に対する日本語での質疑応答能力が高いと言えます。
    Return: { 'score': int, 'rationale': str }
    """

    def build_message(template, question, answer, pred):
        content = template.format(
            input_text=question,
            pred=pred,
            answer=answer,
        )
        message = [{"role": "user", "content": content}]
        return message

    messages = [
        build_message(template, question, answer, pred)
        for question, answer, pred in zip(questions, answers, preds)
    ]

    completion = client.batch_generate_chat_response(
        messages,
        max_tokens=1024,
        temperature=0.0,
    )

    def parse_score(completion):
        # find Score: X
        score = re.search(r"Score: (\d)", completion)
        score = int(score.group(1)) if score else 1
        if score not in [1, 2, 3, 4, 5]:
            raise ValueError("Score Value Error.")

        return {"score": score, "rationale": completion}

    scores = [parse_score(completion) for completion in completion]

    return scores


def test_rouge_ja():
    import pytest

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


if __name__ == "__main__":
    refs = ["晴れている"]
    preds = ["この写真では、晴れた天気が描かれています。"]
    print(rouge_ja(refs, preds))
    print(rouge_ja(["白色"], ["サーフボードは白色です。"]))

    print(rouge_ja(["黒"], ["乗り物の先頭は黒色です。"]))

    from azure_client import OpenAIChatAPI

    client = OpenAIChatAPI()
    print(
        llm_as_a_judge(
            client,
            "質問: {input_text}\n予測: {pred}\n正解: {answer}\n",
            ["これは何色ですか？"],
            ["黒"],
            ["黒色です。"],
        )
    )
