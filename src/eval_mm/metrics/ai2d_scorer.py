import re

from .scorer import Scorer, AggregateOutput
from .scorer_registry import register_scorer
from ._text_utils import strip_reasoning


_CHOICE_LETTER = re.compile(r"\b([A-Da-d])\b")


@register_scorer("ai2d")
class AI2DScorer(Scorer):
    def normalize_answer(self, answer: str) -> str:
        """Normalize reference answer (single letter expected)."""
        return answer.strip().rstrip('.').upper()

    def extract_choice(self, pred: str) -> str | None:
        """Extract a single A-D choice letter from a prediction.

        Handles three cases seen in the wild:
        - "B" — single letter, trivial
        - "B. D" — "label. content" (ai2d option labels contain letters) →
          take the label before the period
        - "<think>...</think>\\n\\nB" — reasoning-model output → strip then extract
        """
        text = strip_reasoning(pred)
        if not text:
            return None
        # Common case: starts with a letter followed by optional punctuation.
        head = text.strip()
        if head and head[0].upper() in {"A", "B", "C", "D"}:
            return head[0].upper()
        # Fallback: find the first standalone A-D letter anywhere.
        m = _CHOICE_LETTER.search(text)
        return m.group(1).upper() if m else None

    def score(self, refs: list[str], preds: list[str]) -> list[int]:
        """Score predictions against references for A-D multiple choice."""
        scores = []
        for ref, pred in zip(refs, preds):
            normalized_ref = self.normalize_answer(ref)
            chosen = self.extract_choice(pred)
            if chosen is not None and chosen == normalized_ref:
                scores.append(1)
            else:
                scores.append(0)
        return scores
    
    def aggregate(self, scores: list[int]) -> AggregateOutput:
        """
        Calculate mean accuracy from scores.
        """
        if len(scores) == 0:
            mean = 0.0
        else:
            mean = sum(scores) / len(scores)
        
        return AggregateOutput(mean, {"accuracy": mean})
