from PIL import Image
from datasets import Dataset, load_dataset

from ..api.registry import register_task
from ..api.task import Task
from eval_mm.metrics import ScorerRegistry
import os


@register_task("jic-vqa")
class JICVQA(Task):
    @staticmethod
    def _prepare_dataset() -> Dataset:
        if not os.path.exists("dataset/jic_vqa.parquet"):
            raise FileNotFoundError(
                "Dataset not found. Please run `scripts/prepare_jic_vqa.py` to prepare the dataset."
            )

        dataset = load_dataset(
            "parquet", data_files="dataset/jic_vqa.parquet", split="train"
        )
        return dataset

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

    def calc_scores(self, preds: list[dict], metric: str) -> list:
        """Calculate scores of each prediction based on the metric."""
        docs = self.dataset
        refs = [doc["answer"] for doc in docs]
        pred_texts = [pred["text"] for pred in preds]
        scorer = ScorerRegistry.get_scorer(metric)
        kwargs = {
            "docs": docs,
            "client": self.client,
            "judge_model": self.config.judge_model,
            "batch_size": self.config.batch_size_for_evaluation,
        }
        return scorer.score(refs, pred_texts, **kwargs)

    def gather_scores(self, scores: list[dict], metric: str):
        scorer = ScorerRegistry.get_scorer(metric)
        kwargs = {"docs": self.dataset}
        return scorer.aggregate(scores, **kwargs)


def test_task():
    from eval_mm.api.task import TaskConfig

    task = JICVQA(TaskConfig())
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), int)
    assert isinstance(task.doc_to_answer(ds[0]), str)
    assert isinstance(task.calc_scores([{"text": "dummy"}], "rougel"), list)
    assert isinstance(task.gather_scores([0.0, 100.0], "rougel"), float)
