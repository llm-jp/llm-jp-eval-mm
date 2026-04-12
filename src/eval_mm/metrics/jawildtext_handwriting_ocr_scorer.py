import re
import unicodedata

from .scorer import AggregateOutput, Scorer
from .scorer_registry import register_scorer


def _normalize_ocr_text(text: str) -> str:
    """Normalize text for CER comparison.

    NFKC normalization, lowercase, newline/spacing handling.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\\n", "\n")
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    cleaned_lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in lines]
    return "\n".join(cleaned_lines).lower()


def _levenshtein(s1: str, s2: str) -> int:
    """Compute edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


@register_scorer("jawildtext-handwriting-ocr")
class JaWildTextHandwritingOCRScorer(Scorer):
    def score(self, refs: list[str], preds: list[str]) -> list[float]:
        scores = []
        for ref, pred in zip(refs, preds):
            ref_norm = _normalize_ocr_text(ref)
            pred_norm = _normalize_ocr_text(pred)
            if not ref_norm and not pred_norm:
                scores.append(1.0)
                continue
            if not ref_norm:
                scores.append(0.0)
                continue
            dist = _levenshtein(ref_norm, pred_norm)
            cer_similarity = max(0.0, 1.0 - dist / len(ref_norm))
            scores.append(cer_similarity)
        return scores

    @staticmethod
    def aggregate(scores: list[float]) -> AggregateOutput:
        if not scores:
            return AggregateOutput(0.0, {"cer_similarity": 0.0})
        mean = sum(scores) / len(scores)
        return AggregateOutput(mean, {"cer_similarity": mean})
