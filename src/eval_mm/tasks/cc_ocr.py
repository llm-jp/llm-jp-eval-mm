from datasets import Dataset, load_dataset
from .task import Task
from PIL import Image
from io import BytesIO
import base64


def base64_to_pil_image(base64_string: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(base64_string)))


class CCOCR(Task):
    """
    The CCOCR class processes the CC-OCR dataset for Japanese samples and provides
    methods to interact with the dataset. It filters the dataset to include only
    entries labeled as "Japanese" and decodes base64-encoded images into PIL Image
    objects for visual processing.
    """

    default_metric = "ccocr"

    @staticmethod
    def _prepare_dataset() -> Dataset:
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

    task = CCOCR(TaskConfig())
    ds = task.dataset
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
    print(ds[0])


if __name__ == "__main__":
    test_task()
