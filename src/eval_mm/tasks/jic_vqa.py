import time
import warnings
from io import BytesIO

import requests
from PIL import Image
from datasets import Dataset, load_dataset
from huggingface_hub import cached_assets_path

from ..api.registry import register_task
from ..api.task import Task
from eval_mm.metrics import ScorerRegistry
from tqdm import tqdm


@register_task("jic-vqa")
class JICVQA(Task):
    @staticmethod
    def _prepare_dataset() -> Dataset:
        cache_dir = cached_assets_path(
            library_name="datasets", namespace="JICVQA", subfolder="download"
        )

        dataset = load_dataset("line-corporation/JIC-VQA")
        input_texts = []
        answers = []
        images = []
        question_ids = []
        domains = []

        domain_dict = {
            "花": "jaflower30",
            "食べ物": "jafood101",
            "ランドマーク": "jalandmark10",
            "施設": "jafacility20",
        }

        def get_domain_from_question(question):
            for keyword, domain in domain_dict.items():
                if keyword in question:
                    return domain

        def download_image(url, image_id):
            # TODO: Multi-threading for faster download
            img_format = url.split(".")[-1]
            image_path = cache_dir / f"{image_id}.{img_format}"
            if image_path.exists():
                return

            max_attempts = 5
            attempt_errors = []
            for _ in range(max_attempts):
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        image.save(image_path)
                        print(f"Downloaded: {image_path}")
                        wait_time = 1.0
                        time.sleep(wait_time)
                        return
                    else:
                        error_msg = f"Status code: {response.status_code}"
                        attempt_errors.append(error_msg)

                except Exception as e:
                    error_msg = f"Exception: {e}"
                    attempt_errors.append(error_msg)

            warnings.warn(
                f"Failed to download {url} after {max_attempts} attempts. Errors: {attempt_errors}"
            )

        # Phase 1: Download all images
        for subset in dataset:
            for entry in tqdm(dataset[subset], desc=f"Downloading {subset} images"):
                url = entry["url"]
                image_id = entry["id"]
                download_image(url, image_id)

        # Phase 2: Load images and populate data structures
        for subset in dataset:
            for entry in dataset[subset]:
                image_id = entry["id"]
                img_format = entry["url"].split(".")[-1]
                image_path = cache_dir / f"{image_id}.{img_format}"

                if not image_path.exists():
                    warnings.warn(f"The image path {image_path} does not exist.")
                    continue
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    print(f"{e} : Failed to open {image_path}")
                images.append(image)
                input_texts.append(entry["question"])
                answers.append(entry["category"])
                question_ids.append(image_id)
                domain = get_domain_from_question(entry["question"])
                domains.append(domain)

        data_dict = {
            "input_text": input_texts,
            "answer": answers,
            "image": images,
            "question_id": question_ids,
            "domain": domains,
        }
        return Dataset.from_dict(data_dict)

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
