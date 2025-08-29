from .scorer import Scorer, AggregateOutput
from .scorer_registry import register_scorer
import re


@register_scorer("substring-match")
class SubstringMatchScorer(Scorer):
    # Word to number mapping
    WORD_TO_NUM = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
        'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
        'million': '1000000', 'billion': '1000000000'
    }
    
    @staticmethod
    def normalize_numbers(text: str) -> str:
        """Convert number words to digits in text."""
        text_lower = text.lower()
        
        # Replace number words with digits
        for word, num in SubstringMatchScorer.WORD_TO_NUM.items():
            # Use word boundaries to avoid partial matches
            text_lower = re.sub(r'\b' + word + r'\b', num, text_lower)
        
        return text_lower
    @staticmethod
    def score(refs: list[str | list[str]], preds: list[str]) -> list[int]:
        scores = []
        for ref, pred in zip(refs, preds):
            # Normalize prediction: lowercase and convert number words
            pred_normalized = SubstringMatchScorer.normalize_numbers(pred.lower())
            
            # Handle case where ref is a list of valid answers
            if isinstance(ref, list):
                # Normalize each reference answer
                refs_normalized = [SubstringMatchScorer.normalize_numbers(r.lower()) for r in ref]
                # Score is 1 if any of the valid answers is in the prediction
                score = int(any(r_norm in pred_normalized for r_norm in refs_normalized))
            else:
                # Handle single answer case
                ref_normalized = SubstringMatchScorer.normalize_numbers(ref.lower())
                score = int(ref_normalized in pred_normalized)
            scores.append(score)
        return scores

    @staticmethod
    def aggregate(scores: list[int]) -> AggregateOutput:
        mean = sum(scores) / len(scores)
        return AggregateOutput(mean, {"substring_match": mean})


def test_substring_match_scorer():
    from .scorer import ScorerConfig

    scorer = SubstringMatchScorer(ScorerConfig())
    refs = ["私は猫です。"]
    preds = ["私は猫です。"]
    scores = scorer.score(refs, preds)
    assert scores == [1]
    refs = ["たかしが公園で遊んでいた。"]
    preds = ["たかしが公園にいたようだ。"]
    scores = scorer.score(refs, preds)
    assert scores == [0]
    refs = ["私は猫です。", "私は犬です。"]
    preds = ["私は犬です。", "私は猫です。"]
    scores = scorer.score(refs, preds)
    assert scores == [0, 0]
    refs = ["池のほとりです。"]
    preds = ["ここは湖の岸です。"]
    scores = scorer.score(refs, preds)
    assert scores == [0]

    output = scorer.aggregate([1, 1, 1, 0])
    assert output.overall_score == 0.75
    assert output.details == {"substring_match": 0.75}
    
    # Test with list of valid answers (like DocVQA)
    refs = [["university of california", "UC"], ["0.28", "0.280"]]
    preds = ["The university of california is great", "The value is 0.280"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1]  # Both should match
    
    refs = [["apple", "orange"], ["cat", "dog"]]
    preds = ["I like bananas", "I have a cat"]
    scores = scorer.score(refs, preds)
    assert scores == [0, 1]  # First no match, second matches "cat"
    
    # Test case-insensitive matching
    refs = ["Hello", ["World", "EARTH"], "Python"]
    preds = ["hello there", "This is the world", "I love PYTHON"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1]  # All should match with case-insensitive
    
    refs = ["Tokyo", ["New York", "NYC"]]
    preds = ["TOKYO is big", "I visited new york"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1]  # Both should match with case-insensitive
    
    # Test number word conversion
    refs = ["2", "5", ["10", "ten"], "20"]
    preds = ["There are two apples", "I have five dollars", "ten items", "twenty people"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1, 1]  # All should match with number conversion
    
    # Test mixed number formats
    refs = ["three", ["4", "four"], "100"]
    preds = ["I see 3 cats", "There are four birds", "one hundred percent"]
    scores = scorer.score(refs, preds)
    assert scores == [1, 1, 1]  # All should match
    
    # Test that partial matches are avoided
    refs = ["one"]
    preds = ["someone is here"]  # "one" in "someone" should not match
    scores = scorer.score(refs, preds)
    assert scores == [0]  # Should not match due to word boundaries
