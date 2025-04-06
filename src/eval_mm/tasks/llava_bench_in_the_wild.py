from datasets import Dataset, load_dataset

from .task import Task
from PIL import Image


class LlavaBenchIntheWild(Task):
    default_metric = "rougel"

    @staticmethod
    def _prepare_dataset() -> Dataset:
        # データセットをロード
        ds = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train")
        ds = ds.rename_column("question", "input_text")
        ds = ds.rename_column("gpt_answer", "answer")
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
        return doc["answer"]


def test_task():
    from eval_mm.tasks.task import TaskConfig

    task = LlavaBenchIntheWild(TaskConfig())
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
