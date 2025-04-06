from datasets import Dataset, load_dataset
import re


from .task import Task
from PIL import Image

# import neologdn FIXME: fix c++12 error when installing neologdn


class JAMultiImageVQA(Task):
    default_metric = "rougel"

    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset("SakanaAI/JA-Multi-Image-VQA", split="test")
        ds = ds.rename_column("question", "input_text")
        ds = ds.map(lambda example, idx: {"question_id": idx}, with_indices=True)
        return ds

    @staticmethod
    def doc_to_text(doc) -> str:
        # delete redundant image tags
        text = re.sub(r"<image> ", "", doc["input_text"])
        return text

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        return doc["images"]

    @staticmethod
    def doc_to_id(doc) -> str:
        return str(doc["question_id"])

    @staticmethod
    def doc_to_answer(doc) -> str:
        return doc["answer"]


def test_task():
    from eval_mm.tasks.task import TaskConfig

    task = JAMultiImageVQA(TaskConfig())
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
