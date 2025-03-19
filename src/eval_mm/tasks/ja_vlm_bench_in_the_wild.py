from datasets import Dataset, load_dataset

from ..api.registry import register_task
from ..api.task import Task
from PIL import Image


@register_task("ja-vlm-bench-in-the-wild")
class JaVLMBenchIntheWild(Task):
    @staticmethod
    def _prepare_dataset() -> Dataset:
        # データセットをロード
        ds = load_dataset("SakanaAI/JA-VLM-Bench-In-the-Wild", split="test")
        ds = ds.rename_column("question", "input_text")
        ds = ds.map(lambda example, idx: {"question_id": idx}, with_indices=True)
        return ds

    @staticmethod
    def doc_to_text(doc) -> str:
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        return [doc["image"]]

    @staticmethod
    def doc_to_id(doc) -> int:
        return doc["question_id"]

    @staticmethod
    def doc_to_answer(doc) -> str:
        return doc["answer"]


def test_task():
    from eval_mm.api.task import TaskConfig

    task = JaVLMBenchIntheWild(TaskConfig())
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), int)
    assert isinstance(task.doc_to_answer(ds[0]), str)
