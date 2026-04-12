import json

from datasets import Dataset, load_dataset
from PIL import Image

from .task import Task, TaskConfig
from .task_registry import register_task

RECEIPT_KIE_PROMPT = (
    "レシート画像からキー情報を抽出し、JSON形式で返してください。 "
    "フィールド: store_name, store_address, receipt_id, date, time, "
    "total_amount, tax_amount, line_items[]. "
    "値は画像の文字をそのまま出力してください（推測・正規化・整形しない）。"
    "無い項目は null にしてください。 "
    'line_items は {"item_name": "", "item_price": "", '
    '"item_quantity": ""} の配列で返してください。'
)

_REPO_ID = "llm-jp/jawildtext"
_CONFIG = "receipt_kie"

_SCALAR_FIELDS = [
    "store_name", "store_address", "receipt_id",
    "date", "time", "total_amount", "tax_amount",
]


def _build_gold_answer(fields: dict) -> str:
    """Build gold answer JSON from the fields struct."""
    if not fields:
        return json.dumps({f: None for f in _SCALAR_FIELDS}, ensure_ascii=False)
    gold: dict = {}
    for field in _SCALAR_FIELDS:
        field_data = fields.get(field, {})
        if isinstance(field_data, dict):
            gold[field] = field_data.get("value")
        else:
            gold[field] = None
    raw_items = fields.get("line_items") or []
    line_items = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        li = {}
        for subfield in ("item_name", "item_price", "item_quantity"):
            sub_data = item.get(subfield, {})
            li[subfield] = sub_data.get("value") if isinstance(sub_data, dict) else None
        line_items.append(li)
    gold["line_items"] = line_items
    return json.dumps(gold, ensure_ascii=False)


def _build_dataset(ds: Dataset) -> Dataset:
    gold_answers = [_build_gold_answer(f) for f in ds["fields"]]
    input_texts = [RECEIPT_KIE_PROMPT] * len(ds)
    question_ids = [str(i) for i in range(len(ds))]
    ds = ds.add_column("gold_answer", gold_answers)
    ds = ds.add_column("input_text", input_texts)
    ds = ds.add_column("question_id", question_ids)
    return ds


@register_task("jawildtext-receipt-kie")
class JaWildTextReceiptKIE(Task):
    default_metric = "jawildtext-receipt-kie"

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
        return doc["gold_answer"]


def test_task():
    task = JaWildTextReceiptKIE(TaskConfig(max_dataset_len=3))
    ds = task.dataset
    assert len(ds) <= 3
    doc = ds[0]
    assert isinstance(JaWildTextReceiptKIE.doc_to_text(doc), str)
    assert isinstance(JaWildTextReceiptKIE.doc_to_visual(doc), list)
    assert isinstance(JaWildTextReceiptKIE.doc_to_visual(doc)[0], Image.Image)
    assert isinstance(JaWildTextReceiptKIE.doc_to_id(doc), str)
    answer = JaWildTextReceiptKIE.doc_to_answer(doc)
    assert isinstance(answer, str)
    parsed = json.loads(answer)
    assert "store_name" in parsed
    assert "line_items" in parsed
