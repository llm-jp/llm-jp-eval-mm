import json
import re
import unicodedata
from collections import Counter

from .scorer import AggregateOutput, Scorer
from .scorer_registry import register_scorer
from ._text_utils import strip_reasoning

_SCALAR_FIELDS = [
    "store_name",
    "store_address",
    "receipt_id",
    "date",
    "time",
    "total_amount",
    "tax_amount",
]


def _extract_json_from_response(text: str) -> dict | None:
    """Extract a JSON object from model response text."""
    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try raw JSON object
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        pass
                    break
    return None


def _normalize_kie_value(value: str) -> str:
    """Normalize a KIE field value for comparison."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.lower().strip()
    text = re.sub(r"[¥￥,]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _flatten_to_tokens(data: dict) -> Counter:
    """Flatten a KIE result dict to 'key:value' token counter for F1."""
    tokens: list[str] = []
    for field in _SCALAR_FIELDS:
        value = data.get(field)
        if value is not None:
            norm = _normalize_kie_value(str(value))
            if norm:
                tokens.append(f"{field}:{norm}")
    line_items = data.get("line_items") or []
    for i, item in enumerate(line_items):
        if not isinstance(item, dict):
            continue
        for subfield in ("item_name", "item_price", "item_quantity"):
            value = item.get(subfield)
            if value is not None:
                norm = _normalize_kie_value(str(value))
                if norm:
                    tokens.append(f"line_items.{subfield}:{norm}")
    return Counter(tokens)


def _token_f1(gold_counter: Counter, pred_counter: Counter) -> dict[str, float]:
    """Calculate token-level precision, recall, and F1."""
    if not gold_counter and not pred_counter:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_counter:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold_counter:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    overlap = gold_counter & pred_counter
    correct = sum(overlap.values())
    precision = correct / sum(pred_counter.values())
    recall = correct / sum(gold_counter.values())
    if precision + recall == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _per_field_accuracy(gold: dict, pred: dict) -> dict[str, float]:
    """Calculate per-field exact match accuracy after normalization."""
    results = {}
    for field in _SCALAR_FIELDS:
        gold_val = _normalize_kie_value(str(gold.get(field, "")))
        pred_val = _normalize_kie_value(str(pred.get(field, "")))
        results[f"field_{field}"] = 1.0 if gold_val == pred_val else 0.0
    return results


@register_scorer("jawildtext-receipt-kie")
class JaWildTextReceiptKIEScorer(Scorer):
    def score(self, refs: list[str], preds: list[str]) -> list[dict]:
        scores = []
        zero_field_acc = {f"field_{f}": 0.0 for f in _SCALAR_FIELDS}
        for ref_json, pred_text in zip(refs, preds):
            gold = json.loads(ref_json)
            pred = _extract_json_from_response(strip_reasoning(pred_text))
            if pred is None:
                scores.append({
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "field_accuracy": zero_field_acc,
                    "parse_error": True,
                })
                continue
            gold_tokens = _flatten_to_tokens(gold)
            pred_tokens = _flatten_to_tokens(pred)
            f1_result = _token_f1(gold_tokens, pred_tokens)
            field_acc = _per_field_accuracy(gold, pred)
            scores.append({
                "f1": f1_result["f1"],
                "precision": f1_result["precision"],
                "recall": f1_result["recall"],
                "field_accuracy": field_acc,
                "parse_error": False,
            })
        return scores

    @staticmethod
    def aggregate(scores: list[dict]) -> AggregateOutput:
        if not scores:
            return AggregateOutput(0.0, {"f1": 0.0})
        f1_scores = [s["f1"] for s in scores]
        mean_f1 = sum(f1_scores) / len(f1_scores)
        mean_precision = sum(s["precision"] for s in scores) / len(scores)
        mean_recall = sum(s["recall"] for s in scores) / len(scores)
        parse_errors = sum(1 for s in scores if s.get("parse_error"))
        # Per-field accuracy (excluding parse errors)
        valid = [s for s in scores if not s.get("parse_error")]
        field_accs = {}
        for field in _SCALAR_FIELDS:
            key = f"field_{field}"
            vals = [s["field_accuracy"][key] for s in valid]
            field_accs[key] = sum(vals) / len(vals) if vals else 0.0
        details = {
            "f1": mean_f1,
            "precision": mean_precision,
            "recall": mean_recall,
            "parse_error_count": parse_errors,
            **field_accs,
        }
        return AggregateOutput(mean_f1, details)
