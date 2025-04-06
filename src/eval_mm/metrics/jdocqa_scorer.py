from eval_mm.metrics.scorer import Scorer, AggregateOutput
from sacrebleu import sentence_bleu
from unicodedata import normalize

ANSWER_TYPE_MAP = {
    "yesno": 0,  # Yes/No questions
    "factoid": 1,  # Factoid questions
    "numerical": 2,  # Numerical questions
    "open-ended": 3,  # Open-ended questions
}

NUM_TO_ANSWER_TYPE = {v: k for k, v in ANSWER_TYPE_MAP.items()}


def jdocqa_normalize(text):
    text = (
        text.replace("です", "")
        .replace("。", "")
        .replace("、", "")
        .replace(" ", "")
        .strip()
    )
    text = normalize("NFKC", text)
    return text


def bleu_ja(refs, pred):
    """Calculate BLEU score for Japanese text. Score is normalized to [0, 1]."""
    bleu_score = sentence_bleu(
        hypothesis=pred,
        references=refs,
        smooth_method="exp",
        smooth_value=0.0,
        tokenize="ja-mecab",
        use_effective_order=False,
        lowercase=False,
    )
    return bleu_score.score / 100


class JDocQAScorer(Scorer):
    def score(self, refs: list[str], preds: list[str]) -> list[int]:
        docs = self.config.docs
        assert docs is not None
        scores = []

        for doc, ref, pred in zip(docs, refs, preds):
            if doc["answer_type"] == ANSWER_TYPE_MAP["open-ended"]:
                scores.append(bleu_ja([ref], pred))
            elif doc["answer_type"] in [
                ANSWER_TYPE_MAP["yesno"],
                ANSWER_TYPE_MAP["factoid"],
                ANSWER_TYPE_MAP["numerical"],
            ]:
                ref = jdocqa_normalize(ref)
                pred = jdocqa_normalize(pred)
                if ref in pred:
                    scores.append(1)
                else:
                    scores.append(0)
            else:
                raise NotImplementedError("Bad answer type.")

        return scores

    def aggregate(self, scores: list[int]) -> AggregateOutput:
        docs = self.config.docs
        assert docs is not None

        # スコア収集用の dict（値はリスト）
        raw_metrics: dict[str, list[float]] = {
            "yesno_exact": [],
            "factoid_exact": [],
            "numerical_exact": [],
            "open-ended_bleu": [],
        }

        for doc, score in zip(docs, scores):
            answer_type = doc["answer_type"]
            if answer_type == ANSWER_TYPE_MAP["open-ended"]:
                raw_metrics["open-ended_bleu"].append(score)
            else:
                key = f"{NUM_TO_ANSWER_TYPE[answer_type]}_exact"
                raw_metrics[key].append(score)

        # 平均値をとって dict[str, float] にする
        metrics: dict[str, float] = {}
        for key, value in raw_metrics.items():
            metrics[key] = sum(value) / len(value) if value else 0.0

        metrics["overall"] = sum(scores) / len(scores) if scores else 0.0

        return AggregateOutput(metrics["overall"], metrics)


def test_jdocqa_scorer():
    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    from .scorer import ScorerConfig

    scorer = JDocQAScorer(ScorerConfig(docs=[{"answer_type": 1}]))
    scores = scorer.score(refs, preds)
    assert scores == [1.0]
    output = scorer.aggregate(scores)
    assert output.overall_score == 1.0
    assert output.details == {
        "factoid_exact": 1.0,
        "yesno_exact": 0,
        "numerical_exact": 0,
        "open-ended_bleu": 0,
        "overall": 1.0,
    }
