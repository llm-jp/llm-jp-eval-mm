from eval_mm.tasks.task import Task
from eval_mm.tasks.task_registry import register_task
from datasets import load_dataset, Dataset
from PIL import Image


@register_task("okvqa")
class OKVQA(Task):
    """OK-VQA task implementation.
    
    OK-VQA (Outside Knowledge Visual Question Answering) requires methods
    which can draw upon outside knowledge to answer questions.
    Multiple valid answers are provided for each question.
    """
    
    def __init__(self, config):
        super().__init__(config)
    
    @staticmethod
    def _prepare_dataset() -> Dataset:
        """Load OK-VQA validation set."""
        # Load the OK-VQA dataset from lmms-lab
        ds = load_dataset("lmms-lab/OK-VQA", split="val2014")
        
        return ds
    
    @staticmethod
    def doc_to_text(doc) -> str:
        """Convert document to text prompt.
        
        OK-VQA is a VQA task that requires external knowledge,
        so we just return the question.
        """
        return doc['question']
    
    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        """Extract image from document."""
        return [doc['image']]
    
    @staticmethod
    def doc_to_id(doc) -> str:
        """Return unique question ID."""
        return str(doc['question_id'])
    
    @staticmethod
    def doc_to_answer(doc) -> list[str]:
        """Return list of valid answers.
        
        OK-VQA provides multiple answers (usually 10) from different annotators.
        We return all unique answers for evaluation with substring-match scorer.
        """
        # Get unique answers while preserving order
        answers = doc['answers']
        unique_answers = []
        seen = set()
        for answer in answers:
            if answer not in seen:
                seen.add(answer)
                unique_answers.append(answer)
        
        return unique_answers


def test_okvqa_task():
    """Test OK-VQA task implementation."""
    from eval_mm.tasks.task import TaskConfig
    
    # Create task instance
    task = OKVQA(TaskConfig(max_dataset_len=10))
    
    # Load dataset
    print("Loading OK-VQA dataset...")
    ds = task.dataset
    print(f"Dataset size: {len(ds)}")
    
    # Test with first example
    example = ds[0]
    print(f"\nFirst example:")
    print(f"  ID: {task.doc_to_id(example)}")
    print(f"  Question: {task.doc_to_text(example)}")
    print(f"  Question Type: {example.get('question_type', 'N/A')}")
    print(f"  Answer Type: {example.get('answer_type', 'N/A')}")
    print(f"  Image: {task.doc_to_visual(example)[0]}")
    print(f"  All answers: {example['answers'][:5]}... (total: {len(example['answers'])})")
    print(f"  Unique answers: {task.doc_to_answer(example)}")
    
    # Verify data types
    assert isinstance(task.doc_to_text(example), str)
    assert isinstance(task.doc_to_visual(example), list)
    assert all(isinstance(img, Image.Image) for img in task.doc_to_visual(example))
    assert isinstance(task.doc_to_id(example), str)
    assert isinstance(task.doc_to_answer(example), list)
    assert all(isinstance(ans, str) for ans in task.doc_to_answer(example))
    
    print("\nAll tests passed!")
