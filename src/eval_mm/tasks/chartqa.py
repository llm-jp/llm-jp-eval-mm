from eval_mm.tasks.task import Task
from eval_mm.tasks.task_registry import register_task
from datasets import load_dataset, Dataset
from PIL import Image


@register_task("chartqa")
class ChartQA(Task):
    """ChartQA task implementation.
    
    ChartQA is a large-scale benchmark for question answering about charts
    with visual and logical reasoning. It covers 9.6K human-written questions
    as well as 23.1K questions generated from human-written chart summaries.
    """
    
    def __init__(self, config):
        super().__init__(config)
    
    def _prepare_dataset(self) -> Dataset:
        """Load ChartQA validation set."""
        # Load the ChartQA dataset from lmms-lab
        ds = load_dataset("lmms-lab/ChartQA", split="test")
        ds = ds.map(lambda example, idx: {"question_id": idx}, with_indices=True)
        return ds

    def _prepare_test_dataset(self) -> Dataset:
        n = getattr(self.config, "max_dataset_len", 10)
        ds = load_dataset("lmms-lab/ChartQA", split=f"test[:{n}]")
        ds = ds.map(lambda example, idx: {"question_id": idx}, with_indices=True)
        return ds
    
    @staticmethod
    def doc_to_text(doc) -> str:
        """Convert document to text prompt.
        
        ChartQA is a QA task, so we just return the question.
        """
        pre_prompt = ""
        question = doc.get('question', '')
        if not question:
            raise ValueError("Document does not contain a valid question.")
        post_prompt = "\nAnswer the question with a single word."
        return f"{pre_prompt}{question}{post_prompt}"

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        """Extract image from document."""
        return [doc['image']]
    
    @staticmethod
    def doc_to_id(doc) -> str:
        """Return unique question ID."""
        # Use the query as ID if no specific ID field
        return str(doc['question_id'])
    
    @staticmethod
    def doc_to_answer(doc) -> list[str]:
        """Return list of valid answers.
        
        ChartQA typically has a single answer but we return it as a list
        for compatibility with substring-match scorer.
        """
        answer = doc.get('label', doc.get('answer', ''))
        if isinstance(answer, list):
            return answer
        return [str(answer)]


def test_chartqa_task():
    """Test ChartQA task implementation."""
    from eval_mm.tasks.task import TaskConfig
    
    # Create task instance
    task = ChartQA(TaskConfig(max_dataset_len=10))
    
    # Load dataset
    print("Loading ChartQA dataset...")
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
