from eval_mm.metrics.scorer import Scorer
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
    bleu_score = sentence_bleu(
        hypothesis=pred,
        references=refs,
        smooth_method="exp",
        smooth_value=0.0,
        tokenize="ja-mecab",
        use_effective_order=False,
        lowercase=False,
    )
    return bleu_score.score


class JDocQAScorer(Scorer):
    @staticmethod
    def score(refs: list[str], preds: list[str], **kwargs) -> list[int]:
        docs = kwargs["docs"]
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

    @staticmethod
    def aggregate(scores: list[int], **kwargs) -> dict:
        docs = kwargs["docs"]
        metrics = {
            "yesno_exact": [],
            "factoid_exact": [],
            "numerical_exact": [],
            "open-ended_bleu": [],
        }
        for doc, score in zip(docs, scores):
            answer_type = doc["answer_type"]
            if answer_type == ANSWER_TYPE_MAP["open-ended"]:
                metrics["open-ended_bleu"].append(score)
            else:
                metrics[f"{NUM_TO_ANSWER_TYPE[answer_type]}_exact"].append(score)

        for key, value in metrics.items():
            if len(value) == 0:
                metrics[key] = 0
                continue
            metrics[key] = sum(value) / len(value)

        return metrics


if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("shunk031/JDocQA", split="test")
    ds = ds.select(range(10))

    ref = ds["answer"][0]
    pred = ds["answer"][0]
    print(ref)
    print(pred)
    print(bleu_ja([ref], pred))
    answer_types = ds["answer_type"]
    answers = ds["answer"]
    print("Original answers")
    for answer_type, answer in zip(answer_types, answers):
        print(NUM_TO_ANSWER_TYPE[answer_type], answer)

    print("JDocQA normalized answers")
    jdocqa_normalize_answers = [jdocqa_normalize(x) for x in ds["answer"]]
    for answer_type, answer in zip(answer_types, jdocqa_normalize_answers):
        print(NUM_TO_ANSWER_TYPE[answer_type], answer)

    scores = JDocQAScorer.score(refs=ds["answer"], preds=ds["answer"], docs=ds)
    print(scores)
    metrics = JDocQAScorer.aggregate(scores, docs=ds)
    print(metrics)
