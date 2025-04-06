# Reference: https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/jmmmu/utils.py

from datasets import (
    Dataset,
    load_dataset,
    concatenate_datasets,
    get_dataset_config_names,
)

from .task import Task

import ast
import re
from PIL import Image


MULTI_CHOICE_PROMPT = (
    "与えられた選択肢の中から最も適切な回答のアルファベットを直接記入してください。"
)
OPEN_ENDED_PROMPT = "質問に対する回答を単語や短いフレーズで記入してください。"


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


def jmmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    question = replace_images_tokens(question)  # TODO: check if this is necessary
    return question


def jmmmu_doc_to_visual(doc):
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


class JMMMU(Task):
    default_metric = "jmmmu"

    @staticmethod
    def _prepare_dataset() -> Dataset:
        configs = get_dataset_config_names("JMMMU/JMMMU")
        datasets = [
            load_dataset("JMMMU/JMMMU", name=subject, split="test")
            for subject in configs
        ]
        dataset = concatenate_datasets(datasets)
        dataset = dataset.map(
            lambda x: {
                "input_text": jmmmu_doc_to_text(x),
                "question_id": x["id"],
                "answer": x["answer"],
            }
        )
        return dataset

    @staticmethod
    def doc_to_text(doc) -> str:
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        return jmmmu_doc_to_visual(doc)

    @staticmethod
    def doc_to_id(doc) -> str:
        return str(doc["question_id"])

    @staticmethod
    def doc_to_answer(doc) -> str:
        return doc["answer"]


def test_task():
    from eval_mm.tasks.task import TaskConfig

    task = JMMMU(TaskConfig())
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
