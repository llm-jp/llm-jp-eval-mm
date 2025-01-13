# Reference: https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/mmmu/utils.py

from datasets import (
    Dataset,
    load_dataset,
    concatenate_datasets,
    get_dataset_config_names,
)

from ..api.registry import register_task
from ..api.task import Task

import ast
import re
from eval_mm.metrics import ScorerRegistry

MULTI_CHOICE_PROMPT = "Answer with the option's letter from the given choices directly."
OPEN_ENDED_PROMPT = "Answer the question using a single word or phrase."


def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join(
        [
            f"{option_letter}. {option}"
            for option_letter, option in zip(option_letters, options)
        ]
    )
    return choices_str


def construct_prompt(doc):
    question = doc["question"]
    question = question.replace("<image1>", "<image 1>")
    if doc["question_type"] == "multiple-choice":
        # Weirdly, data["options"] is a string in MMMU Huggingface dataset
        parsed_options = parse_options(ast.literal_eval(doc["options"]))
        # parsed_options already prepends a newline so no need to add space here
        question = f"{question}\n{parsed_options}\n\n{MULTI_CHOICE_PROMPT}"
    else:
        question = f"{question}\n\n{OPEN_ENDED_PROMPT}"
    return question


def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    question = replace_images_tokens(question)  # TODO: check if this is necessary
    return question


def mmmu_doc_to_visual(doc):
    prompt = construct_prompt(doc)

    image_tokens = re.findall(r"<image \d+>", prompt)
    # Remove <> and  swap space as _
    image_tokens = sorted(
        list(
            set(
                [
                    image_token.strip("<>").replace(" ", "_")
                    for image_token in image_tokens
                ]
            )
        )
    )
    visual = [doc[image_token].convert("RGB") for image_token in image_tokens]
    return visual


@register_task("mmmu")
class MMMU(Task):
    @staticmethod
    def _prepare_dataset() -> Dataset:
        configs = get_dataset_config_names("MMMU/MMMU")
        dataset = None
        for subject in configs:
            if dataset is None:
                dataset = load_dataset("MMMU/MMMU", name=subject, split="validation")
            else:
                dataset = concatenate_datasets(
                    [
                        dataset,
                        load_dataset("MMMU/MMMU", name=subject, split="validation"),
                    ]
                )
        dataset = dataset.map(
            lambda x: {
                "input_text": mmmu_doc_to_text(x),
                "question_id": x["id"],
                "answer": x["answer"],
            }
        )
        return dataset

    @staticmethod
    def doc_to_text(doc):
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc):
        return mmmu_doc_to_visual(doc)

    @staticmethod
    def doc_to_id(doc):
        return doc["question_id"]

    @staticmethod
    def doc_to_answer(doc):
        return doc["answer"]

    def calc_scores(self, preds: list, metric: str) -> list:
        """Calculate scores of each prediction based on the metric."""
        docs = self.dataset
        refs = [doc["answer"] for doc in docs]
        pred_texts = [pred["text"] for pred in preds]
        scorer = ScorerRegistry.get_scorer(metric)
        kwargs = {
            "docs": docs,
            "client": self.client,
            "batch_size": self.config.batch_size_for_evaluation,
            "judge_model": self.config.judge_model,
        }
        return scorer.score(refs, pred_texts, **kwargs)

    def gather_scores(self, scores: list[dict], metric: str) -> dict:
        """Gather scores of each prediction based on the metric."""
        kwargs = {"docs": self.dataset}
        scorer = ScorerRegistry.get_scorer(metric)
        return scorer.aggregate(scores, **kwargs)
