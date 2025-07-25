from .scorer import Scorer, AggregateOutput
from .scorer_registry import register_scorer


@register_scorer("blink", "BLINK", "BLINKScorer")
class BLINKScorer(Scorer):
    """Scorer for BLINK Benchmark multiple-choice questions.
    
    BLINK uses answer format like '(A)', '(B)', etc., but we normalize
    to just the letter for consistency with other multiple-choice tasks.
    """
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer string by:
        - Stripping whitespace
        - Removing trailing periods first (to handle cases like "(A).")
        - Removing parentheses
        - Converting to uppercase
        
        Examples:
        - "(A)" -> "A"
        - " a. " -> "A"
        - "B" -> "B"
        - "(D)." -> "D"
        """
        # Strip whitespace
        answer = answer.strip()
        
        # Remove trailing periods first (this handles "(A)." -> "(A)")
        answer = answer.rstrip('.')
        
        # Remove parentheses if present
        if answer.startswith('(') and answer.endswith(')'):
            answer = answer[1:-1]
        
        # Strip again in case there were spaces inside parentheses
        answer = answer.strip()
        
        # Convert to uppercase
        answer = answer.upper()
        
        return answer
    
    def score(self, refs: list[str], preds: list[str]) -> list[int]:
        """
        Score predictions against references for BLINK multiple-choice questions.
        """
        scores = []
        for ref, pred in zip(refs, preds):
            # Normalize both reference and prediction
            normalized_ref = self.normalize_answer(ref)
            normalized_pred = self.normalize_answer(pred)
            
            # Check if normalized prediction is valid and matches reference
            if normalized_pred in ['A', 'B', 'C', 'D'] and normalized_ref == normalized_pred:
                scores.append(1)
            else:
                scores.append(0)
        
        return scores
    
    def aggregate(self, scores: list[int]) -> AggregateOutput:
        """
        Calculate overall accuracy and per-config accuracy if config info is available.
        """
        if len(scores) == 0:
            return AggregateOutput(0.0, {"accuracy": 0.0})
        
        overall_accuracy = sum(scores) / len(scores)
        
        # For now, just return overall accuracy
        # In the future, we could add per-config breakdown if needed
        details = {
            "accuracy": overall_accuracy,
            "total_samples": len(scores),
            "correct_samples": sum(scores)
        }
        
        return AggregateOutput(overall_accuracy, details)


def test_blink_scorer():
    """Test BLINK scorer functionality."""
    from .scorer import ScorerConfig
    
    scorer = BLINKScorer(ScorerConfig())
    
    # Test basic functionality with BLINK format
    refs = ["A", "B", "C", "D"]
    preds = ["(A)", "(B)", "(C)", "(D)"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1], f"Expected all correct, got {scores}"
    
    # Test normalization - parentheses removal
    refs = ["A", "B", "C", "D"]
    preds = ["(A)", "B", "(C)", "D"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1], f"Expected all correct, got {scores}"
    
    # Test normalization - spaces
    refs = ["A", "B", "C", "D"]
    preds = [" A ", "B ", " C", "D"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1], f"Expected all correct, got {scores}"
    
    # Test normalization - periods
    refs = ["A", "B", "C", "D"]
    preds = ["A.", "B.", "C.", "D."]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1], f"Expected all correct, got {scores}"
    
    # Test normalization - lowercase
    refs = ["A", "B", "C", "D"]
    preds = ["a", "b", "c", "d"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1], f"Expected all correct, got {scores}"
    
    # Test mixed normalization
    refs = ["A", "B", "C", "D"]
    preds = [" (a) ", "b.", " C ", "(D)."]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1], f"Expected all correct, got {scores}"
    
    # Test incorrect answers
    refs = ["A", "B", "C", "D"]
    preds = ["B", "A", "D", "C"]
    scores = scorer.score(refs, preds)
    assert scores == [0, 0, 0, 0], f"Expected all incorrect, got {scores}"
    
    # Test invalid answers
    refs = ["A", "B", "C", "D"]
    preds = ["E", "1", "X", ""]
    scores = scorer.score(refs, preds)
    assert scores == [0, 0, 0, 0], f"Expected all incorrect, got {scores}"
    
    # Test aggregation
    output = scorer.aggregate([1, 1, 1, 0])
    assert output.overall_score == 0.75, f"Expected 0.75, got {output.overall_score}"
    assert output.details["accuracy"] == 0.75
    assert output.details["total_samples"] == 4
    assert output.details["correct_samples"] == 3
    
    # Test empty scores
    output = scorer.aggregate([])
    assert output.overall_score == 0.0
    assert output.details["accuracy"] == 0.0
    
    print("All tests passed!")