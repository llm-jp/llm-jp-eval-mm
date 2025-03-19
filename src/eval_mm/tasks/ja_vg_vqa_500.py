from datasets import Dataset, concatenate_datasets, load_dataset

from ..api.registry import register_task
from ..api.task import Task
from PIL import Image


@register_task("ja-vg-vqa-500")
class JaVGVQA500(Task):
    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset("SakanaAI/JA-VG-VQA-500", split="test")

        def flatten_sample(sample):
            dataset = {
                "image_id": [sample["image_id"] for _ in sample["qas"]],
                "image": [sample["image"] for _ in sample["qas"]],
                "qa_id": [qa["qa_id"] for qa in sample["qas"]],
                "question": [qa["question"] for qa in sample["qas"]],
                "answer": [qa["answer"] for qa in sample["qas"]],
            }
            return Dataset.from_dict(dataset)

        fragments = []
        for i, sample in enumerate(ds):
            data_fragment = flatten_sample(sample)
            fragments.append(data_fragment)

        ds = concatenate_datasets(fragments)
        ds = ds.rename_column("question", "input_text")
        ds = ds.rename_column("qa_id", "question_id")

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

    task = JaVGVQA500(TaskConfig())
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), int)
    assert isinstance(task.doc_to_answer(ds[0]), str)
