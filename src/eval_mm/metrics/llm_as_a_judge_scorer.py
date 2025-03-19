from eval_mm.metrics.scorer import Scorer
from tqdm import tqdm
import re


INSTRUCTION = """
You are an expert evaluator. You are given a (Question, Answer, Prediction) triplet. Your task is to evaluate how well the Prediction aligns with the Answer in the context of the Question.

Please assign a score from 1 to 5 based on the following criteria:

5: Excellent — The Prediction fully matches the Answer with complete correctness and relevance.
4: Good — The Prediction is mostly correct with only minor inaccuracies or missing details.
3: Fair — The Prediction is partially correct but contains noticeable errors or missing key points.
2: Poor — The Prediction is mostly incorrect or irrelevant, but there are small fragments of correctness.
1: Very Poor — The Prediction is completely incorrect or irrelevant.
Output only the score (an integer from 1 to 5). Do not add any explanation.

Triplet:
Question: {Question}
Answer: {Answer}
Prediction: {Prediction}

Your Score:
"""


class LlmAsaJudgeScorer(Scorer):
    @staticmethod
    def score(
        refs,
        preds,
        **kwargs,
    ):
        client = kwargs["client"]
        model_name = kwargs["judge_model"]
        batch_size = kwargs["batch_size"]
        docs = kwargs["docs"]
        questions = docs["input_text"]

        def build_message(question: str, answer: str, pred: str):
            content = INSTRUCTION.format(
                Question=question, Answer=answer, Prediction=pred
            )
            message = [
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": content},
            ]
            return message

        messages = [
            build_message(question, answer, pred)
            for question, answer, pred in zip(questions, refs, preds)
        ]
        messages_list = [
            messages[i : i + batch_size] for i in range(0, len(messages), batch_size)
        ]
        completion = []
        for ms in tqdm(messages_list, desc="Evaluating LLM as a Judge"):
            completion.extend(
                client.batch_generate_chat_response(
                    ms, max_tokens=1024, temperature=0.0, seed=0, model_name=model_name
                )
            )

        scores = []
        for i, c in enumerate(completion):
            print(c)
            score = re.search(r"\d", c)
            if score:
                scores.append(int(score.group()))
            else:
                scores.append(0)
        # if preds is empty, return 0 (TODO: this process should be done before calling llm)
        # for i, pred in enumerate(preds):
        #     if pred == "":
        #         scores[i] = 0
        return scores

    def aggregate(scores: list, **kwargs) -> float:
        return sum([score for score in scores]) / len(scores)


def test_llm_as_a_judge_scorer():
    from eval_mm.utils.azure_client import MockChatAPI, OpenAIChatAPI

    batch_size = 1
    questions = ["What is the capital of Japan?", "What is the capital of France?"]
    answers = ["Tokyo", "Paris"]
    preds = ["", ""]
    scores = LlmAsaJudgeScorer.score(
        answers,
        preds,
        docs={"input_text": questions},
        client=MockChatAPI(),
        judge_model="moch",
        batch_size=batch_size,
    )
    assert scores == [0, 0]
    scores = LlmAsaJudgeScorer.aggregate(scores)
    assert scores == 0.0

    import os

    if os.getenv("AZURE_OPENAI_KEY"):
        questions = ["What is the capital of Japan?", "What is the capital of France?"]
        answers = ["Tokyo", "Paris"]
        preds = ["Tokyo", "Paris"]
        batch_size = 1
        model_name = "gpt-4o-mini-2024-07-18"
        scores = LlmAsaJudgeScorer.score(
            answers,
            preds,
            docs={"input_text": questions},
            client=OpenAIChatAPI(),
            judge_model=model_name,
            batch_size=batch_size,
        )
        assert scores == [5, 5]

        questions = [
            "前年の合計所得が33万円以下の世帯の場合、軽減の割合は何割ですか。\n解答は数量のみで答えてください。",
            "3つのラベルはどのような違いがありますか。\n解答は自由に記述してください。",
            "国内の日本語教育の概要において、外国人に対する日本語教育の現状として、日本語教師数の数や学習者の数はどうなっていますか。\n解答は自由に記述してください。",
        ]
        answers = [
            "7割です",
            "下部に文字がプリントされているかどうか、プリントの内容が「CARBONNEUTRAL」または「CERTIFIEDMODEL」かの違いがあります。",
            "国内の日本語教育の概要において、外国人に対する日本語教育の現状として、日本語教師数の数は34392人、学習者の数は139613人となっています。",
        ]

        preds = [
            "7割",
            "",
            "日本語教師数の数は34392人、学習者の数は139613人となっています。",
        ]
        model_name = "gpt-4o-2024-05-13"
        scores = LlmAsaJudgeScorer.score(
            answers,
            preds,
            docs={"input_text": questions},
            client=OpenAIChatAPI(),
            judge_model=model_name,
            batch_size=batch_size,
        )
        assert scores == [5, 1, 5]
