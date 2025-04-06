from eval_mm.tasks.task import Task
from datasets import load_dataset, Dataset
from PIL import Image


class MNIST(Task):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset("ylecun/mnist", split="test")
        ds = ds.map(lambda example, idx: {"question_id": idx}, with_indices=True)
        return ds

    @staticmethod
    def doc_to_text(doc) -> str:
        return "画像に写っている数字は何ですか？ 数字のみを出力してください。"

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        return [doc["image"]]

    @staticmethod
    def doc_to_id(doc) -> str:
        return str(doc["question_id"])

    @staticmethod
    def doc_to_answer(doc) -> str:
        return str(doc["label"])


def test_task():
    from eval_mm.tasks.task import TaskConfig

    task = MNIST(TaskConfig())
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
