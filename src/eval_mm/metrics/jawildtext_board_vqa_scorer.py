import re

from tqdm import tqdm

from .scorer import AggregateOutput, Scorer
from .scorer_registry import register_scorer

GRADER_TEMPLATE = """\
あなたは回答の正誤を判定する専門家です。
以下の[質問]に対する[正解]と、[モデルの回答]を比較し、正誤を判定してください。

[質問]: {question}
[正解]: {gold_answer}
[モデルの回答]: {prediction}

判定ルール:
- 表記の揺れは許容してください（例: 2羽 vs 二羽, 8,000 vs 8000）
- 数字の全角半角の違いは許容してください
- 正解の内容を含んでいれば、追加の文脈があっても正解としてください
- 正解と意味的に同じ回答であれば、言い回しが異なっていても正解としてください
- 単位が質問から自明な場合、省略されていても正解としてください

以下の形式のみで回答してください:
correct: yes
または
correct: no
"""


def _extract_boxed_answer(text: str) -> str:
    r"""Extract answer from \boxed{...} in model response.

    Returns the content of the last \boxed{} found,
    or the full text stripped if none found.
    """
    # Handle nested braces by counting depth
    results = []
    i = 0
    prefix = "\\boxed{"
    while i < len(text):
        pos = text.find(prefix, i)
        if pos == -1:
            break
        start = pos + len(prefix)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start : j - 1].strip())
        i = j
    return results[-1] if results else text.strip()


@register_scorer("jawildtext-board-vqa")
class JaWildTextBoardVQAScorer(Scorer):
    def score(self, refs: list[str], preds: list[str]) -> list[float]:
        client = self.config.client
        model_name = self.config.judge_model
        batch_size = self.config.batch_size
        docs = self.config.docs
        questions = docs["input_text"]

        # Extract boxed answers from predictions
        extracted = [_extract_boxed_answer(p) for p in preds]

        # Build grader messages, skipping empty predictions
        empty_indices: set[int] = {
            i for i, pred in enumerate(extracted) if pred == ""
        }
        messages = []
        for i, (question, gold, pred) in enumerate(
            zip(questions, refs, extracted)
        ):
            if i in empty_indices:
                continue
            content = GRADER_TEMPLATE.format(
                question=question, gold_answer=gold, prediction=pred
            )
            messages.append([
                {"role": "system", "content": "あなたは回答の正誤を判定する専門家です。"},
                {"role": "user", "content": content},
            ])

        # Batch LLM judge calls
        messages_batches = [
            messages[i : i + batch_size]
            for i in range(0, len(messages), batch_size)
        ]
        completions: list[str] = []
        for batch in tqdm(messages_batches, desc="Evaluating JaWildText Board VQA"):
            completions.extend(
                client.batch_generate_chat_response(
                    batch,
                    max_tokens=256,
                    temperature=0.0,
                    seed=0,
                    model_name=model_name,
                )
            )

        # Merge scores
        scores: list[float] = []
        comp_iter = iter(completions)
        for i in range(len(preds)):
            if i in empty_indices:
                scores.append(0.0)
            else:
                response = next(comp_iter)
                match = re.search(r"correct\s*:\s*(yes|no)", response, re.IGNORECASE)
                scores.append(1.0 if match and match.group(1).lower() == "yes" else 0.0)
        return scores

    @staticmethod
    def aggregate(scores: list[float]) -> AggregateOutput:
        if not scores:
            return AggregateOutput(0.0, {"accuracy": 0.0})
        mean = sum(scores) / len(scores)
        return AggregateOutput(mean, {"accuracy": mean})
