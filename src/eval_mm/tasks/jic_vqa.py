import time
from pathlib import Path
from PIL import Image, ImageFile
import requests
from io import BytesIO
from datasets import Dataset, load_dataset

from ..api.registry import register_task
from ..api.task import Task
from eval_mm.metrics import ScorerRegistry

@register_task("jic_vqa")
class JICVQA(Task):
    @staticmethod
    def _prepare_dataset() -> Dataset:
        wait_time = 1.0
        cache_dir = Path("/tmp/image_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_dataset("line-corporation/JIC-VQA")
        input_texts = []
        answers = []
        images = []
        question_ids = []
        domains = []

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None

        domain_dict = {
            "花": "jaflower30",
            "食べ物": "jafood101",
            "ランドマーク": "jalandmark10",
            "施設": "jafacility20"
        }

        def get_domain_from_question(question):
            for keyword, domain in domain_dict.items():
                if keyword in question:
                    return domain

        def download_image(url, image_id):
            img_format = url.split('.')[-1]
            image_path = cache_dir / f"{image_id}.{img_format}"
            if image_path.exists():
                return

            try:
                response = requests.get(url)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    image.save(image_path)
                    print(f"Downloaded: {image_path}")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to download image: {url}, skipped")
            except Exception as e:
                print(f"{e} : Failed to download {url}, skipped")

        # Phase 1: Download all images
        for subset in dataset:
            for entry in dataset[subset]:
                url = entry["url"]
                image_id = entry["id"]
                download_image(url, image_id)

        # Phase 2: Load images and populate data structures
        for subset in dataset:
            for entry in dataset[subset]:
                image_id = entry["id"]
                img_format = entry["url"].split('.')[-1]
                image_path = cache_dir / f"{image_id}.{img_format}"

                if not image_path.exists():
                    continue  
                
                try:
                    image = Image.open(image_path)
                    images.append(image)
                    input_texts.append(entry["question"])
                    answers.append(entry["category"])
                    question_ids.append(image_id)
                    domain = get_domain_from_question(entry["question"])
                    domains.append(domain)
                except Exception as e:
                    print(f"{e} : Failed to open {image_path}")

        data_dict = {
            "input_text": input_texts,
            "answer": answers,
            "image": images,
            "question_id": question_ids,
            "domain": domains
        }
        return Dataset.from_dict(data_dict)

    @staticmethod
    def doc_to_text(doc):
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc):
        return doc["image"]

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

    def gather_scores(self, scores: list[dict], metric: str):
        scorer = ScorerRegistry.get_scorer(metric)
        kwargs = {"docs": self.dataset}
        return scorer.aggregate(scores, **kwargs)
