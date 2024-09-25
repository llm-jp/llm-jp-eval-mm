from datasets import Dataset, load_dataset

from ..api.registry import register_task
from ..api.task import Task
from ..utils.metrics import sacrebleu_ja_sent

answer_type_map = {
    "yesno": "1",  # Yes/No questions
    "factoid": "2",  # Factoid questions
    "numerical": "3",  # Numerical questions
    "open-ended": "4",  # Open-ended questions
}


def pdf_to_image(pdf_path):
    # TODO: Implement pdf to image conversion
    image = None
    return image


@register_task("jdocqa")
class JDocQA(Task):
    def __init__(self, config=None) -> None:
        super().__init__(config)

    def prepare_task(self, config) -> None:
        self._dataset = load_dataset("shunk031/JDocQA", split="test")

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
            if doc["answer_type"] == answer_type_map["open-ended"]:
                # sacrebleu
                refs = [doc["answer"]]
                scores.append(sacrebleu_ja_sent(refs, pred["text"]))
            else:
                # === Exact match with Simple Rules ===
                # TODO: Implement exact match with simple rules
                scores.append(0.0)

        eval_results = docs
        eval_results["score"] = scores

        return eval_results

    def compute_metrics(self, preds, model_id=None, batch_size=1):
        """Process the results of the model.
        Args:
            jsonl_path: jsonl_path
            preds: [pred]
            model_id: openai api's model name (default: "gpt-4o-mini-2024-07-18")
        Return:
            metrics: a dictionary with key: {
                "1_yesno_exact": float,
                "2_factoid_exact": float,
                "3_numerical_exact": float,
                "4_open-ended_bleu": float,
            }
            eval_results: a list of dictionaries with keys:
                { TODO: fill this }
        """
        eval_results = self.evaluate(self.dataset, preds)
        metrics = {
            "1_yesno_exact": [],
            "2_factoid_exact": [],
            "3_numerical_exact": [],
            "4_open-ended_bleu": [],
        }

        # Collect scores
        for eval_result in eval_results:
            answer_type = eval_result["answer_type"]
            if eval_result["answer_type"] == self.answer_type_map["open-ended"]:
                suffix = "bleu"
            else:
                suffix = "exact"
            metrics_key = f"{answer_type}_{answer_type_map[answer_type]}_{suffix}"
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
                {
                    "question_id": str, "text": str,
                }
            eval_results:
                list of dictionaries with keys:
                {
                    "score", # etc.
                }
        Return:
            dictonaries with keys:
            {
                    "question_id": str,
                    "text": str,
                    "score": float,
            }
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
