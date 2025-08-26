from eval_mm.tasks.task import Task
from eval_mm.tasks.task_registry import register_task
from datasets import load_dataset, Dataset
from PIL import Image
import re


@register_task("mathvista")
class MathVista(Task):
    """MathVista task implementation.
    
    MathVista evaluates mathematical reasoning in visual contexts.
    It includes both multiple-choice and free-form questions with various answer types.
    """
    
    def __init__(self, config):
        super().__init__(config)
    
    @staticmethod
    def _prepare_dataset() -> Dataset:
        """Load MathVista testmini set."""
        # Load the MathVista dataset from AI4Math
        ds = load_dataset("AI4Math/MathVista", split="testmini")
        
        return ds
    
    @staticmethod
    def doc_to_text(doc) -> str:
        """Convert document to text prompt.
        
        Format the prompt based on question type.
        """
        question = doc['question']
        question_type = doc.get('question_type', '')
        
        if question_type == 'multi_choice' and doc.get('choices'):
            # Format multiple choice question with choices
            choices_str = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(doc['choices'])])
            prompt = f"{question}\n\nChoices:\n{choices_str}\n\nAnswer with the letter of the correct choice."
        else:
            # Free-form question
            prompt = f"{question}\n\nProvide a direct answer."
        
        return prompt
    
    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        """Extract decoded image from document.
        
        MathVista provides a decoded_image field that contains
        the actual PIL Image object.
        """
        decoded_image = doc.get('decoded_image')
        if decoded_image is None:
            raise ValueError("No decoded_image found in document")
        
        return [decoded_image]
    
    @staticmethod
    def doc_to_id(doc) -> str:
        """Return unique problem ID."""
        return str(doc['pid'])
    
    @staticmethod
    def doc_to_answer(doc) -> str:
        """Return the answer.
        
        MathVista answers are already in string format.
        For multiple-choice questions, the answer is one of the choices.
        For free-form questions, the answer is a number or text.
        """
        return str(doc['answer'])
    
    @staticmethod
    def doc_to_metadata(doc) -> dict:
        """Return metadata for scorer.
        
        This includes all information needed for proper scoring.
        """
        return {
            'question_type': doc.get('question_type', 'unknown'),
            'answer_type': doc.get('answer_type', 'unknown'),
            'choices': doc.get('choices', []),
            'precision': doc.get('precision', None),
            'unit': doc.get('unit', None),
        }


def test_mathvista_task():
    """Test MathVista task implementation."""
    from eval_mm.tasks.task import TaskConfig
    
    # Create task instance
    task = MathVista(TaskConfig(max_dataset_len=10))
    
    # Load dataset
    print("Loading MathVista dataset...")
    ds = task.dataset
    print(f"Dataset size: {len(ds)}")
    
    # Test with first example
    example = ds[0]
    print(f"\nFirst example:")
    print(f"  ID: {task.doc_to_id(example)}")
    print(f"  Prompt: {task.doc_to_text(example)[:100]}...")
    print(f"  Answer: {task.doc_to_answer(example)}")
    print(f"  Metadata: {task.doc_to_metadata(example)}")
    print(f"  Image: {task.doc_to_visual(example)[0]}")
    
    # Verify data types
    assert isinstance(task.doc_to_text(example), str)
    assert isinstance(task.doc_to_visual(example), list)
    assert all(isinstance(img, Image.Image) for img in task.doc_to_visual(example))
    assert isinstance(task.doc_to_id(example), str)
    assert isinstance(task.doc_to_answer(example), str)
    assert isinstance(task.doc_to_metadata(example), dict)
    
    print("\nAll tests passed!")
