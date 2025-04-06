from datasets import load_dataset, Dataset

from .task import Task
from PIL import Image


class JapaneseHeronBench(Task):
    default_metric = "heron-bench"

    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset("Silviase/Japanese-Heron-Bench", split="train")
        ds = ds.rename_column("text", "input_text")
        return ds

    @staticmethod
    def doc_to_text(doc) -> str:
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        return [doc["image"]]

    @staticmethod
    def doc_to_id(doc) -> str:
        return str(doc["question_id"])

    @staticmethod
    def doc_to_answer(doc) -> str:
        return doc["answer"]["gpt-4-0125-preview"]


def test_task():
    from eval_mm.tasks.task import TaskConfig

    task = JapaneseHeronBench(TaskConfig())
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
