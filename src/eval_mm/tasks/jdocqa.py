from datasets import Dataset, load_dataset
from pdf2image import convert_from_path

from ..api.registry import register_task
from ..api.task import Task
from eval_mm.metrics import ScorerRegistry

import aiohttp
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images


def get_elements_from_index(indices_str, array):
    try:
        indices = [int(x.strip()) - 1 for x in indices_str.split(",")]
        elements = [array[i] for i in indices if 0 <= i < len(array)]
        return elements
    except ValueError:
        print("The string doesn't seem to have numbers or commas in the right places.")
        return None  # Or maybe an empty list, depending on how you wanna handle it
    except IndexError:
        print("Out of bounds error!")
        return None  # Same, an empty list or special value could work


@register_task("jdocqa")
class JDocQA(Task):
    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset(
            "shunk031/JDocQA",
            split="test",
            rename_pdf_category=True,
            trust_remote_code=True,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
            },
        )
        ds = ds.rename_column("question", "input_text")
        ds = ds.map(lambda example, idx: {"question_id": idx}, with_indices=True)
        keep_columns = [
            "input_text",
            "pdf_filepath",
            "question_page_number",
            "question_id",
            "answer",
            "answer_type",
        ]
        ds = ds.remove_columns(
            [col for col in ds.column_names if col not in keep_columns]
        )
        return ds

    @staticmethod
    def doc_to_text(doc):
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc):
        images_all = pdf_to_images(doc["pdf_filepath"])
        images = get_elements_from_index(doc["question_page_number"], images_all)
        return images

    @staticmethod
    def doc_to_id(doc):
        return doc["question_id"]

    @staticmethod
    def doc_to_answer(doc):
        return doc["answer"]

    def calc_scores(self, preds: list, metric: str) -> list:
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

    def gather_scores(self, scores: list[dict], metric: str) -> dict:
        kwargs = {"docs": self.dataset}
        scorer = ScorerRegistry.get_scorer(metric)
        return scorer.aggregate(scores, **kwargs)
