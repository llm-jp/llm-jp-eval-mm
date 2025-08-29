from datasets import Dataset, load_dataset
from .task import Task
from .task_registry import register_task
from PIL import Image
from io import BytesIO
import base64
import os


def base64_to_pil_image(base64_string: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(base64_string)))


@register_task("cc-ocr")
class CCOCR(Task):
    """
    The CCOCR class processes the CC-OCR dataset for Japanese samples and provides
    methods to interact with the dataset. It filters the dataset to include only
    entries labeled as "Japanese" and decodes base64-encoded images into PIL Image
    objects for visual processing.
    """

    default_metric = "ccocr"

    def _prepare_dataset(self) -> Dataset:
        ds = load_dataset("wulipc/CC-OCR", "multi_lan_ocr", split="test")
        ds = ds.filter(lambda example: example["l2-category"] == "Japanese")
        ds = ds.map(
            lambda x, idx: {
                "index": str(idx),
                "question_id": str(idx),
                "question": x["question"],
                "answer": x["answer"],
                "input_text": x["question"],
                "image": x["image"],
            },
            with_indices=True,
        )
        return ds

    def _prepare_test_dataset(self) -> Dataset:
        # Stream to collect first N Japanese samples without downloading full split
        n = getattr(self.config, "max_dataset_len", 10)
        stream = load_dataset("wulipc/CC-OCR", "multi_lan_ocr", split="test", streaming=True)
        buf = {
            "index": [],
            "question_id": [],
            "question": [],
            "answer": [],
            "input_text": [],
            "image": [],
        }
        count = 0
        for ex in stream:
            if ex.get("l2-category") == "Japanese":
                buf["index"].append(str(count))
                buf["question_id"].append(str(count))
                buf["question"].append(ex["question"])
                buf["answer"].append(ex["answer"])
                buf["input_text"].append(ex["question"])
                buf["image"].append(ex["image"])
                count += 1
                if count >= n:
                    break
        return Dataset.from_dict(buf)

    @staticmethod
    def doc_to_text(doc) -> str:
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        image = base64_to_pil_image(doc["image"])
        return [image]

    @staticmethod
    def doc_to_id(doc) -> str:
        return str(doc["question_id"])

    @staticmethod
    def doc_to_answer(doc) -> str:
        return doc["answer"]


def test_task():
    from eval_mm.tasks.task import TaskConfig

    # Limit dataset size in tests to reduce runtime
    task = CCOCR(TaskConfig(max_dataset_len=10))
    ds = task.dataset
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
    # Avoid printing full sample to prevent unnecessary I/O


if __name__ == "__main__":
    test_task()
