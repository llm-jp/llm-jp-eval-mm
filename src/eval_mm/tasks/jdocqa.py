from datasets import Dataset, load_dataset

from .task import Task

from PIL import Image


class JDocQA(Task):
    default_metric = "jdocqa"

    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset(
            "speed/JDocQA",
            split="test",
        )
        ds = ds.rename_column("question", "input_text")
        return ds

    @staticmethod
    def doc_to_text(doc) -> str:
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        images = []
        for column in ["image_0", "image_1", "image_2", "image_3"]:
            if doc[column] is not None:
                images.append(doc[column])
        return images

    @staticmethod
    def doc_to_id(doc) -> str:
        return str(doc["question_id"])

    @staticmethod
    def doc_to_answer(doc) -> str:
        return doc["answer"]


def test_task():
    from eval_mm.tasks.task import TaskConfig

    task = JDocQA(TaskConfig())
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
