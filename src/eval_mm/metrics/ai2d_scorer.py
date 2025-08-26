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


def test_ai2d_scorer():
    from .scorer import ScorerConfig
    
    scorer = AI2DScorer(ScorerConfig())
    
    # Test basic functionality
    refs = ["A", "B", "C", "D"]
    preds = ["A", "B", "C", "D"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1]
    
    # Test normalization - spaces
    refs = ["A", "B", "C", "D"]
    preds = [" A ", "B ", " C", "D"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1]
    
    # Test normalization - periods
    refs = ["A", "B", "C", "D"]
    preds = ["A.", "B.", "C.", "D."]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1]
    
    # Test normalization - lowercase
    refs = ["A", "B", "C", "D"]
    preds = ["a", "b", "c", "d"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1]
    
    # Test mixed normalization
    refs = ["A", "B", "C", "D"]
    preds = [" a. ", "b.", " C ", "d"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1]
    
    # Test incorrect answers
    refs = ["A", "B", "C", "D"]
    preds = ["B", "A", "D", "C"]
    scores = scorer.score(refs, preds)
    assert scores == [0, 0, 0, 0]
    
    # Test invalid answers
    refs = ["A", "B", "C", "D"]
    preds = ["E", "1", "X", ""]
    scores = scorer.score(refs, preds)
    assert scores == [0, 0, 0, 0]
    
    # Test aggregation
    output = scorer.aggregate([1, 1, 1, 0])
    assert output.overall_score == 0.75
    assert output.details == {"accuracy": 0.75}
    
    output = scorer.aggregate([1, 0, 0, 0])
    assert output.overall_score == 0.25
    assert output.details == {"accuracy": 0.25}
    
    # Test empty scores
    output = scorer.aggregate([])
    assert output.overall_score == 0.0
    assert output.details == {"accuracy": 0.0}
    
    print("All tests passed!")
