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
