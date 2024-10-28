from datasets import Dataset, load_dataset

from ..api.registry import register_task
from ..api.task import Task
from ..utils.metrics import bleu_ja

ANSWER_TYPE_MAP = {
    "yesno": 0,  # Yes/No questions
    "factoid": 1,  # Factoid questions
    "numerical": 2,  # Numerical questions
    "open-ended": 3,  # Open-ended questions
}

NUM_TO_ANSWER_TYPE = {v: k for k, v in ANSWER_TYPE_MAP.items()}


def pdf_to_image(pdf_path):
    # TODO: Implement pdf to image conversion
    image = None
    return image


def jdocqa_normalize(text):
    text = text.replace("です", "").replace("。", "").replace("、", "").strip()
    return text


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
        )

        # rename columns
        if self._dataset is not None:
            self._dataset = self._dataset.rename_column("question", "input_text")
            self._dataset = self._dataset.map(
                lambda example, idx: {"question_id": idx}, with_indices=True
            )
        else:
            raise ValueError("Dataset is None, cannot rename column.")

        # check dataset's columns and types of each column
        if self._dataset is not None:
            print(self._dataset.column_names)
            print(self._dataset.features)

    def doc_to_text(self, doc):
        return doc["input_text"]

    def doc_to_visual(self, doc):
        # Read pdf from path
        filepath = doc["pdf_filepath"]
        image = pdf_to_image(filepath)
        return image

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
        for doc, pred in zip(docs, preds):
            print("==== answer type ====")
            print(doc["answer_type"], NUM_TO_ANSWER_TYPE[doc["answer_type"]])
            print(doc["answer"], type(doc["answer"]))
            print(doc["pdf_filepath"], type(doc["pdf_filepath"]))

            if doc["answer_type"] == ANSWER_TYPE_MAP["open-ended"]:
                refs = [doc["answer"]]
                scores.append(bleu_ja(refs, pred["text"]))
            else:
                # === Exact match with Simple Rules ===
                # TODO: Implement exact match with simple rules
                scores.append(0.0)
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
    results = task.format_result(preds, eval_results)
    # print(results)
    print(metrics)
