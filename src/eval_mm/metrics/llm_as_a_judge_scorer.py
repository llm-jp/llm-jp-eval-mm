from .scorer import Scorer, AggregateOutput
from .scorer_registry import register_scorer
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


@register_scorer("llm-as-a-judge")
class LlmAsaJudgeScorer(Scorer):
    def score(
        self,
        refs,
        preds,
    ):
        client = self.config.client
        model_name = self.config.judge_model
        batch_size = self.config.batch_size
        docs = self.config.docs
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

        # Track indices of empty predictions so we can skip them in the LLM call
        # and directly assign a score of 0.
        empty_indices: set[int] = {i for i, pred in enumerate(preds) if pred == ""}

        messages = [
            build_message(question, answer, pred)
            for i, (question, answer, pred) in enumerate(zip(questions, refs, preds))
            if i not in empty_indices
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

        # Merge LLM scores with 0s for empty predictions
        scores: list[int] = []
        comp_iter = iter(completion)
        for i in range(len(preds)):
            if i in empty_indices:
                scores.append(0)
            else:
                c = next(comp_iter)
                match = re.search(r"\d", c)
                scores.append(int(match.group()) if match else 0)
        return scores

    @staticmethod
    def aggregate(scores: list) -> AggregateOutput:
        mean = sum(scores) / len(scores)
        return AggregateOutput(mean, {"llm_as_a_judge": mean})
