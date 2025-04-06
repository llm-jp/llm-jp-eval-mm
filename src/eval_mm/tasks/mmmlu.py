from datasets import load_dataset, Dataset

from .task import Task
from PIL import Image


class MMMLU(Task):
    default_metric = "exact-match"

    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset("openai/MMMLU", "JA_JP", split="test")

        # ['Unnamed: 0', 'Question', 'A', 'B', 'C', 'D', 'Answer', 'Subject'],
        def build_prompt(example):
            return f"{example['Question']} A: {example['A']} B: {example['B']} C: {example['C']} D: {example['D']}. Output only the letter of the correct answer. Answer:"

        ds = ds.rename_column("Answer", "answer")
        ds = ds.rename_column("Unnamed: 0", "question_id")
        ds = ds.map(
            lambda example: {"input_text": build_prompt(example)},
            remove_columns=["Question", "A", "B", "C", "D", "Subject"],
        )

        return ds

    @staticmethod
    def doc_to_text(doc) -> str:
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        return []

    @staticmethod
    def doc_to_id(doc) -> str:
        return str(doc["question_id"])

    @staticmethod
    def doc_to_answer(doc) -> str:
        return doc["answer"]


def test_task():
    from .task import TaskConfig

    task = MMMLU(TaskConfig())
    ds = task.dataset
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
