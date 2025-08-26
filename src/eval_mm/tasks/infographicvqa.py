from eval_mm.tasks.task import Task
from eval_mm.tasks.task_registry import register_task
from datasets import load_dataset, Dataset
from PIL import Image


@register_task("infographicvqa")
class InfographicVQA(Task):
    """InfographicVQA task implementation.
    
    InfographicVQA is a VQA dataset for understanding infographic images.
    It's a subset of the DocVQA dataset focusing on infographics.
    Multiple valid answers are provided for each question.
    """
    
    def __init__(self, config):
        super().__init__(config)
    
    def _prepare_dataset(self) -> Dataset:
        """Load InfographicVQA validation set."""
        # Load the InfographicVQA config from lmms-lab/DocVQA dataset
        ds = load_dataset(
            "lmms-lab/DocVQA",
            "InfographicVQA",
            split=self._maybe_slice_split("validation"),
        )
        
        # Rename questionId to question_id for consistency
        ds = ds.rename_column("questionId", "question_id")
        
        return ds
    
    @staticmethod
    def doc_to_text(doc) -> str:
        """Convert document to text prompt.
        
        InfographicVQA is an extractive QA task, so we just return the question.
        """
        post_prompt = "\nAnswer the question using a single word or phrase."
        return f"{doc['question']}{post_prompt}"
    
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
        
        InfographicVQA provides multiple valid answers for each question.
        We return all of them for evaluation with substring-match scorer.
        """
        return doc['answers']


def test_infographicvqa_task():
    """Test InfographicVQA task implementation."""
    from eval_mm.tasks.task import TaskConfig
    
    # Create task instance
    task = InfographicVQA(TaskConfig(max_dataset_len=10))
    
    # Load dataset
    print("Loading InfographicVQA dataset...")
    ds = task.dataset
    print(f"Dataset size: {len(ds)}")
    
    # Test with first example
    example = ds[0]
    print(f"\nFirst example:")
    print(f"  ID: {task.doc_to_id(example)}")
    print(f"  Question: {task.doc_to_text(example)}")
    print(f"  Image: {task.doc_to_visual(example)[0]}")
    print(f"  Valid answers: {task.doc_to_answer(example)}")
    
    # Verify data types
    assert isinstance(task.doc_to_text(example), str)
    assert isinstance(task.doc_to_visual(example), list)
    assert all(isinstance(img, Image.Image) for img in task.doc_to_visual(example))
    assert isinstance(task.doc_to_id(example), str)
    assert isinstance(task.doc_to_answer(example), list)
    assert all(isinstance(ans, str) for ans in task.doc_to_answer(example))
    
    print("\nAll tests passed!")
