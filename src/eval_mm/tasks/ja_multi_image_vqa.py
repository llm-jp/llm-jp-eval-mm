from datasets import Dataset, load_dataset
import re


from .task import Task
from .task_registry import register_task
from PIL import Image

# neologdn provides superior Japanese text normalization (e.g. full/half-width
# unification, prolonged sound mark normalization) but requires a C++12 toolchain
# that may not be available in all build environments.  When it is missing we fall
# back to unicodedata.normalize("NFKC", ...), which covers the most important
# full-width / half-width conversions.
try:
    import neologdn  # noqa: F401
except ImportError:
    neologdn = None


@register_task("ja-multi-image-vqa")
class JAMultiImageVQA(Task):
    default_metric = "rougel"

    def _prepare_dataset(self) -> Dataset:
        ds = load_dataset("SakanaAI/JA-Multi-Image-VQA", split="test")
        ds = ds.rename_column("question", "input_text")
        ds = ds.map(lambda example, idx: {"question_id": idx}, with_indices=True)
        return ds

    def _prepare_test_dataset(self) -> Dataset:
        n = getattr(self.config, "max_dataset_len", 10)
        ds = load_dataset("SakanaAI/JA-Multi-Image-VQA", split=f"test[:{n}]")
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

    task = JAMultiImageVQA(TaskConfig(max_dataset_len=10))
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
