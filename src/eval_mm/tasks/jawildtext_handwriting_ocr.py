from datasets import Dataset, load_dataset
from PIL import Image

from .task import Task, TaskConfig
from .task_registry import register_task

HANDWRITING_OCR_PROMPT = (
    "OCR:\n画像内の文字をすべて読んでください。 "
    "改行されている部分には必ず\\nを挿入してください。"
)

_REPO_ID = "llm-jp/jawildtext"
_CONFIG = "handwriting_ocr"


def _centroid(ann: dict) -> tuple[float, float]:
    coords = ann["polygon"]
    return (
        sum(pt[0] for pt in coords) / len(coords),
        sum(pt[1] for pt in coords) / len(coords),
    )


def _is_tall(ann: dict) -> bool:
    xs = [pt[0] for pt in ann["polygon"]]
    ys = [pt[1] for pt in ann["polygon"]]
    return max(ys) - min(ys) > max(xs) - min(xs)


def _build_reference_from_polygons(polygons: list[dict]) -> str:
    """Build reference text from polygon annotations in reading order.

    The page direction is taken from the line geometry (vertical when most
    line polygons are taller than wide), not from the per-polygon
    `direction` label, which disagrees with the image on a few pages.
    Horizontal pages: lines top to bottom, then left to right.
    Vertical pages: columns right to left, then top to bottom.
    Lines are joined with newlines.
    """
    anns = [ann for ann in polygons or [] if ann.get("text") and ann.get("polygon")]
    if not anns:
        return ""

    vertical = sum(_is_tall(ann) for ann in anns) > len(anns) / 2

    def reading_order_key(ann: dict) -> tuple[float, float]:
        x, y = _centroid(ann)
        return (-x, y) if vertical else (y, x)

    return "\n".join(ann["text"] for ann in sorted(anns, key=reading_order_key))


def _build_dataset(ds: Dataset) -> Dataset:
    ocr_refs = [_build_reference_from_polygons(p) for p in ds["polygons"]]
    input_texts = [HANDWRITING_OCR_PROMPT] * len(ds)
    question_ids = [str(i) for i in range(len(ds))]
    ds = ds.add_column("ocr_reference", ocr_refs)
    ds = ds.add_column("input_text", input_texts)
    ds = ds.add_column("question_id", question_ids)
    return ds


@register_task("jawildtext-handwriting-ocr")
class JaWildTextHandwritingOCR(Task):
    default_metric = "jawildtext-handwriting-ocr"

    def _prepare_dataset(self) -> Dataset:
        ds = load_dataset(_REPO_ID, _CONFIG, split="train")
        return _build_dataset(ds)

    def _prepare_test_dataset(self) -> Dataset:
        n = self.config.max_dataset_len or 10
        ds = load_dataset(_REPO_ID, _CONFIG, split=f"train[:{n}]")
        return _build_dataset(ds)

    @staticmethod
    def doc_to_text(doc) -> str:
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        return [doc["image"]]

    @staticmethod
    def doc_to_id(doc) -> str:
        return doc["question_id"]

    @staticmethod
    def doc_to_answer(doc) -> str:
        return doc["ocr_reference"]


def test_task():
    task = JaWildTextHandwritingOCR(TaskConfig(max_dataset_len=3))
    ds = task.dataset
    assert len(ds) <= 3
    doc = ds[0]
    assert isinstance(JaWildTextHandwritingOCR.doc_to_text(doc), str)
    assert isinstance(JaWildTextHandwritingOCR.doc_to_visual(doc), list)
    assert isinstance(JaWildTextHandwritingOCR.doc_to_visual(doc)[0], Image.Image)
    assert isinstance(JaWildTextHandwritingOCR.doc_to_id(doc), str)
    assert isinstance(JaWildTextHandwritingOCR.doc_to_answer(doc), str)
    assert len(JaWildTextHandwritingOCR.doc_to_answer(doc)) > 0
