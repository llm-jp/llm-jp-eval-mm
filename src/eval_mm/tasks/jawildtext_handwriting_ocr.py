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


def _build_reference_from_polygons(polygons: list[dict]) -> str:
    """Build reference text from polygon annotations.

    Sorts annotations top-to-bottom, left-to-right by centroid,
    then joins text with newlines.
    """
    if not polygons:
        return ""

    def centroid_sort_key(ann: dict) -> tuple[float, float]:
        coords = ann.get("polygon", [])
        if not coords:
            return (0.0, 0.0)
        ys = [pt[1] for pt in coords]
        xs = [pt[0] for pt in coords]
        return (sum(ys) / len(ys), sum(xs) / len(xs))

    sorted_anns = sorted(polygons, key=centroid_sort_key)
    texts = [ann["text"] for ann in sorted_anns if ann.get("text")]
    return "\n".join(texts)


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
