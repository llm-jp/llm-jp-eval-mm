from datasets import Dataset, load_dataset
from pdf2image import convert_from_path

from ..api.registry import register_task
from ..api.task import Task
from ..utils.metrics import bleu_ja

import aiohttp
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ANSWER_TYPE_MAP = {
    "yesno": 0,  # Yes/No questions
    "factoid": 1,  # Factoid questions
    "numerical": 2,  # Numerical questions
    "open-ended": 3,  # Open-ended questions
}

NUM_TO_ANSWER_TYPE = {v: k for k, v in ANSWER_TYPE_MAP.items()}


def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images


def jdocqa_normalize(text):
    text = text.replace("です", "").replace("。", "").replace("、", "").strip()
    return text


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
    def __init__(self, config=None) -> None:
        super().__init__(config)

    def prepare_task(self, config) -> None:
        self._dataset = load_dataset(
            "shunk031/JDocQA",
            split="test",
            rename_pdf_category=True,
            trust_remote_code=True,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
            },
        )

        # rename columns
        if self._dataset is not None:
            self._dataset = self._dataset.rename_column("question", "input_text")
            self._dataset = self._dataset.map(
                lambda example, idx: {"question_id": idx}, with_indices=True
            )
        else:
            raise ValueError("Dataset is None, cannot rename column.")

        # TODO: When the PR is closing, remove this.
        # sample first 30 examples
        self._dataset = self._dataset.select(range(100))

    def doc_to_text(self, doc):
        return jdocqa_normalize(doc["input_text"])

    def doc_to_visual(self, doc):
        images_all = pdf_to_images(doc["pdf_filepath"])
        images = get_elements_from_index(doc["question_page_number"], images_all)
        return images

    def doc_to_id(self, doc):
        return doc["question_id"]

    def process_pred(self, doc, pred):
        processed = doc
        processed["pred"] = pred
        return processed

    def evaluate(self, docs: list, preds: list) -> list[dict]:
        """Evaluate batch prediction.
        Args:
        doc : list of instance of the eval dataset
        pred : list of dict with keys: { 'question_id', 'text' }
        Returns:
        eval_results: list of dictionary with keys:
            { 'input_text', 'pred', 'question_id', 'answer', 'score' }
        """
        assert len(docs) == len(preds), "Length of docs and preds must be equal."
        assert all(
            [
                doc["question_id"] == pred["question_id"]
                for doc, pred in zip(docs, preds)
            ]
        ), "Question IDs must be the same."

        scores = []

        # TMP;
        idx = 0

        for doc, pred in zip(docs, preds):
            answer = doc["answer"]

            if doc["answer_type"] == ANSWER_TYPE_MAP["open-ended"]:
                refs = [doc["answer"]]
                scores.append(bleu_ja(refs, pred["text"]))

            # 著者実装と異なる可能性がある． 注意して利用する．
            # TODO: Implement Unanswerable Questions
            elif doc["answer_type"] in [
                ANSWER_TYPE_MAP["yesno"],
                ANSWER_TYPE_MAP["factoid"],
                ANSWER_TYPE_MAP["numerical"],
            ]:
                if answer in pred["text"]:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            else:
                raise NotImplementedError("Bad answer type.")

            question = self.doc_to_text(doc)
            answer = doc["answer"]
            images = self.doc_to_visual(doc)

            print(f"Idx: {idx}")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            for i, image in enumerate(images):
                image.save(
                    f"/home/silviase/llmjp/llm-jp-eval-multimodal/tmp/jdocqa/image_{idx}_{i}.jpg"
                )

            idx += 1

        eval_results = []
        for doc, pred, score in zip(docs, preds, scores):
            eval_result = doc
            eval_result["pred"] = pred["text"]
            eval_result["score"] = score
            eval_results.append(eval_result)

        return eval_results

    def compute_metrics(self, preds, model_id=None, batch_size=1):
        """Process the results of the model.
        Args:
            jsonl_path: jsonl_path
            preds: [pred]
            model_id: openai api's model name (default: "gpt-4o-mini-2024-07-18")
        Return:
            metrics: a dictionary with key: {
                "yesno_exact": float,
                "factoid_exact": float,
                "numerical_exact": float,
                "open-ended_bleu": float,
            }
            eval_results: a list of dictionaries with keys:
                { TODO: fill this }
        """
        eval_results = self.evaluate(self.dataset, preds)
        metrics = {
            "yesno_exact": [],
            "factoid_exact": [],
            "numerical_exact": [],
            "open-ended_bleu": [],
        }

        # Collect scores
        for eval_result in eval_results:
            answer_type = eval_result["answer_type"]
            if eval_result["answer_type"] == ANSWER_TYPE_MAP["open-ended"]:
                suffix = "bleu"
            else:
                suffix = "exact"
            metrics_key = f"{NUM_TO_ANSWER_TYPE[answer_type]}_{suffix}"
            metrics[metrics_key].append(eval_result["score"])

        # Compute average
        for key, value in metrics.items():
            if len(value) == 0:
                raise ValueError(f"No data for {key}")
            metrics[key] = sum(value) / len(value)

        return metrics, eval_results

    def format_result(self, preds: list[dict], eval_results: list[dict]) -> list[dict]:
        """Format the result of the model.
        Args:
            preds:
                list of dictionaries with keys:
                    - "question_id": str,
                    - "text": str,
            eval_results:
                list of dictionaries with keys:
                    - "score", # etc.
        Return:
            results:
                list of dictonaries with keys:
                    - "question_id": str,
                    - "text": str,
                    - "score": float,
        """
        assert len(preds) == len(
            eval_results
        ), "Length of preds and eval_results must be equal."
        results = []
        for pred, eval_result in zip(preds, eval_results):
            result = {
                "question_id": pred["question_id"],
                "text": pred["text"],
                "score": eval_result["score"],
            }
            results.append(result)
        return results


if __name__ == "__main__":
    task = JDocQA()
    config = None
    task.prepare_task(config)
    docs = task.dataset

    preds = [{"question_id": doc["question_id"], "text": doc["answer"]} for doc in docs]
    metrics, eval_results = task.compute_metrics(preds)

    preds_bad = [
        {"question_id": doc["question_id"], "text": "これは間違いです"} for doc in docs
    ]
    metrics_bad, eval_results_bad = task.compute_metrics(preds_bad)

    print(metrics, metrics_bad)
