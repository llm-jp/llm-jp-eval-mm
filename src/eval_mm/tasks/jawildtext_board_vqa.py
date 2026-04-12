from datasets import Dataset, load_dataset
from PIL import Image

from .task import Task, TaskConfig
from .task_registry import register_task

BOARD_VQA_PROMPT_SUFFIX = (
    "\n画像を参照して回答してください。推論過程は出力しても構いませんが、"
    "最終回答は必ず \\boxed{...} で囲み、ボックス内には最終回答のみを1つだけ記載してください。"
)

_REPO_ID = "llm-jp/jawildtext"
_CONFIG = "board_vqa"


def _build_dataset(ds: Dataset) -> Dataset:
    input_texts = [(q or "") + BOARD_VQA_PROMPT_SUFFIX for q in ds["question"]]
    question_ids = [str(i) for i in range(len(ds))]
    ds = ds.add_column("input_text", input_texts)
    ds = ds.add_column("question_id", question_ids)
    return ds


@register_task("jawildtext-board-vqa")
class JaWildTextBoardVQA(Task):
    default_metric = "jawildtext-board-vqa"

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
        return doc["answer"] or ""


def test_task():
    task = JaWildTextBoardVQA(TaskConfig(max_dataset_len=3))
    ds = task.dataset
    assert len(ds) <= 3
    doc = ds[0]
    assert isinstance(JaWildTextBoardVQA.doc_to_text(doc), str)
    assert isinstance(JaWildTextBoardVQA.doc_to_visual(doc), list)
    assert isinstance(JaWildTextBoardVQA.doc_to_visual(doc)[0], Image.Image)
    assert isinstance(JaWildTextBoardVQA.doc_to_id(doc), str)
    assert isinstance(JaWildTextBoardVQA.doc_to_answer(doc), str)
