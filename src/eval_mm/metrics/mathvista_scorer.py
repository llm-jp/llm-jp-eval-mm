import re
from .scorer import Scorer, AggregateOutput
from .scorer_registry import register_scorer


@register_scorer("mathvista")
class MathVistaScorer(Scorer):
    """Scorer for MathVista dataset based on lmms-eval implementation.
    
    Handles answer extraction and normalization for:
    1. Multiple-choice questions
    2. Free-form questions (numeric and text)
    """
    
    def extract_answer(self, response: str, question_type: str, choices: list[str] = None) -> str:
        """Extract answer from model response.
        
        Based on the lmms-eval MathVista implementation.
        """
        response = response.strip()
        
        # For multiple choice questions, try to extract letter choice
        if question_type == 'multi_choice' and choices:
            # Pattern 1: "The answer is A" or "The answer is (A)"
            patterns = [
                r'[Tt]he answer is[\s]*\(?([A-Za-z])\)?',
                r'[Aa]nswer:[\s]*\(?([A-Za-z])\)?',
                r'\b([A-Za-z])\b[\s]*(?:is|would be|should be)[\s]*(?:the|my)[\s]*(?:answer|choice)',
                r'(?:choose|select|pick)[\s]*\(?([A-Za-z])\)?',
                r'^[\s]*\(?([A-Za-z])\)?[\s]*$',  # Just the letter
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    letter = match.group(1).upper()
                    if letter in 'ABCDEFGHIJ'[:len(choices)]:
                        return letter
            
            # If no clear pattern, look for the first occurrence of a valid letter
            for i, letter in enumerate('ABCDEFGHIJ'[:len(choices)]):
                if letter in response.upper():
                    return letter
        
        # For free-form questions, extract numeric or text answer
        else:
            # Try to extract numeric answer
            # Pattern for numbers including decimals, negative, and scientific notation
            number_patterns = [
                r'[Tt]he answer is[\s]*:?[\s]*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)',
                r'[Aa]nswer:[\s]*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)',
                r'=[\s]*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)',
                r'(?:is|equals|equal to)[\s]*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)',
            ]
            
            for pattern in number_patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group(1)
            
            # If no clear pattern, look for numbers in the response
            # First try to find numbers with thousands separators
            comma_numbers = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?', response)
            if comma_numbers:
                # Remove commas and return the last one
                return comma_numbers[-1].replace(',', '')
            
            # Then look for regular numbers
            all_numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', response)
            if all_numbers:
                # Filter out years and other unlikely answers
                valid_numbers = []
                for num in all_numbers:
                    try:
                        val = float(num)
                        # Skip years and very large numbers unless they seem intentional
                        if not (1900 <= val <= 2100 and len(num) == 4):
                            valid_numbers.append(num)
                    except:
                        pass
                
                if valid_numbers:
                    return valid_numbers[-1]  # Return the last valid number
        
        # If no answer can be extracted, return the full response (truncated)
        return response[:50] if len(response) > 50 else response
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        # Convert to lowercase
        answer = answer.lower()
        
        # Remove common punctuation
        answer = answer.replace(',', '')
        answer = answer.replace('$', '')
        answer = answer.replace('%', '')
        answer = answer.strip('.')
        
        return answer
    
    def compare_answers(self, pred: str, ref: str, answer_type: str, precision: float = None) -> bool:
        """Compare predicted and reference answers based on answer type."""
        # Normalize both answers
        pred_norm = self.normalize_answer(pred)
        ref_norm = self.normalize_answer(ref)
        
        # For numeric answers
        if answer_type in ['integer', 'float']:
            try:
                pred_num = float(pred_norm)
                ref_num = float(ref_norm)
                
                if answer_type == 'integer':
                    # For integers, check exact match after rounding
                    return round(pred_num) == round(ref_num)
                elif precision is not None and precision > 0:
                    # For floats with precision, compare with tolerance
                    tolerance = 10 ** (-int(precision))
                    return abs(pred_num - ref_num) < tolerance
                else:
                    # For floats without precision, use relative tolerance
                    return abs(pred_num - ref_num) < 1e-5 * max(abs(ref_num), 1)
                    
            except ValueError:
                # If not numeric, fall back to string comparison
                pass
        
        # For text answers (including multiple choice letters)
        return pred_norm == ref_norm
    
    def score_single(self, pred: str, ref: str, metadata: dict) -> int:
        """Score a single prediction."""
        question_type = metadata.get('question_type', 'unknown')
        answer_type = metadata.get('answer_type', 'text')
        choices = metadata.get('choices', [])
        precision = metadata.get('precision')
        
        # Extract answer from prediction
        extracted_answer = self.extract_answer(pred, question_type, choices)
        
        # For multiple choice, convert letter to actual answer
        if question_type == 'multi_choice' and choices and len(extracted_answer) == 1:
            letter_idx = ord(extracted_answer.upper()) - ord('A')
            if 0 <= letter_idx < len(choices):
                extracted_answer = choices[letter_idx]
        
        # Compare answers
        is_correct = self.compare_answers(extracted_answer, ref, answer_type, precision)
        
        return 1 if is_correct else 0
    
    def score(self, refs: list[str], preds: list[str], metadata_list: list[dict] | None = None) -> list[int]:
        """Score predictions against references."""
        if metadata_list is None:
            # If no metadata provided, treat as text comparison
            metadata_list = [{'answer_type': 'text'}] * len(refs)
        
        scores = []
        for ref, pred, metadata in zip(refs, preds, metadata_list):
            score = self.score_single(pred, ref, metadata)
            scores.append(score)
        
        return scores
    
    def aggregate(self, scores: list[int]) -> AggregateOutput:
        """Calculate mean accuracy from scores."""
        if len(scores) == 0:
            mean = 0.0
        else:
            mean = sum(scores) / len(scores)
        
        return AggregateOutput(mean, {"accuracy": mean})


def test_mathvista_scorer():
    from .scorer import ScorerConfig
    
    scorer = MathVistaScorer(ScorerConfig())
    
    # Test answer extraction
    
    # Multiple choice
    response = "Looking at the graph, I can see that the value is 5. The answer is B."
    extracted = scorer.extract_answer(response, 'multi_choice', ['3', '5', '7', '9'])
    assert extracted == 'B', f"Expected 'B', got '{extracted}'"
    
    # Numeric
    response = "After calculating, the answer is 42.5"
    extracted = scorer.extract_answer(response, 'free_form')
    assert extracted == '42.5', f"Expected '42.5', got '{extracted}'"
    
    # Test scoring
    
    # Multiple choice correct
    refs = ["5"]
    preds = ["The answer is B"]
    metadata = [{"question_type": "multi_choice", "answer_type": "text", "choices": ["3", "5", "7", "9"]}]
    scores = scorer.score(refs, preds, metadata)
    assert scores == [1], f"Expected [1], got {scores}"
    
    # Numeric with precision
    refs = ["3.14159"]
    preds = ["The value is approximately 3.142"]
    metadata = [{"question_type": "free_form", "answer_type": "float", "precision": 2.0}]
    scores = scorer.score(refs, preds, metadata)
    assert scores == [1], f"Expected [1], got {scores}"
    
    # Integer
    refs = ["1000"]
    preds = ["There are 1,000 items in total"]
    metadata = [{"question_type": "free_form", "answer_type": "integer"}]
    scores = scorer.score(refs, preds, metadata)
    assert scores == [1], f"Expected [1], got {scores}"
    
    # No prints to keep tests quiet
