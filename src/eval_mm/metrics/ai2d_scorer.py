from .scorer import Scorer, AggregateOutput
from .scorer_registry import register_scorer


@register_scorer("ai2d")
class AI2DScorer(Scorer):
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer string by:
        - Stripping whitespace
        - Removing trailing periods
        - Converting to uppercase
        """
        return answer.strip().rstrip('.').upper()
    
    def score(self, refs: list[str], preds: list[str]) -> list[int]:
        """
        Score predictions against references for multiple-choice questions.
        Normalizes both refs and preds before comparison.
        """
        scores = []
        for ref, pred in zip(refs, preds):
            # Normalize both reference and prediction
            normalized_ref = self.normalize_answer(ref)
            normalized_pred = self.normalize_answer(pred)
            
            # Check if normalized prediction matches reference and is valid (A, B, C, or D)
            if normalized_pred in ['A', 'B', 'C', 'D'] and normalized_ref == normalized_pred:
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
