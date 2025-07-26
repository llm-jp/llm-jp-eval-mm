import base64
import io
from eval_mm.tasks.task import Task
from eval_mm.tasks.task_registry import register_task
from datasets import load_dataset, Dataset
from PIL import Image


@register_task("chartqapro", "ChartQAPro", "chart-qa-pro")
class ChartQAPro(Task):
    """ChartQAPro task implementation.
    
    ChartQAPro is a more diverse and challenging benchmark for chart question answering.
    It includes 1,341 charts from 157 diverse sources, spanning various chart types
    including infographics and dashboards, with 1,948 questions of various types.
    """
    
    def __init__(self, config):
        super().__init__(config)
    
    @staticmethod
    def _prepare_dataset() -> Dataset:
        """Load ChartQAPro test set."""
        # Load the ChartQAPro dataset from ahmed-masry
        ds = load_dataset("ahmed-masry/ChartQAPro", split="test")
        
        return ds
    
    @staticmethod
    def doc_to_text(doc) -> str:
        """Convert document to text prompt.
        
        ChartQAPro is a QA task, so we return the question.
        """
        return doc['Question']
    
    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        """Extract and decode image from document.
        
        ChartQAPro stores images as binary/base64 encoded data.
        """
        image_data = doc['image']
        
        # If image_data is bytes, decode directly
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        # If image_data is base64 string, decode first
        elif isinstance(image_data, str):
            # Remove potential data URL prefix
            if image_data.startswith('data:image'):
                image_data = image_data.split(',', 1)[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        # If image_data is already a PIL Image
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError(f"Unexpected image data type: {type(image_data)}")
        
        return [image]
    
    @staticmethod
    def doc_to_id(doc) -> str:
        """Return unique question ID."""
        # Use question and year as ID components
        question = doc['Question']
        year = doc.get('Year', ['unknown'])[0] if doc.get('Year') else 'unknown'
        return f"{year}_{hash(question)}"
    
    @staticmethod
    def doc_to_answer(doc) -> list[str]:
        """Return list of valid answers.
        
        ChartQAPro may have multiple valid answers.
        """
        answer = doc.get('Answer', [])
        if isinstance(answer, list):
            # Filter out empty strings and ensure all are strings
            return [str(a) for a in answer if a]
        elif isinstance(answer, str):
            return [answer] if answer else []
        else:
            return [str(answer)] if answer else []


def test_chartqapro_task():
    """Test ChartQAPro task implementation."""
    from eval_mm.tasks.task import TaskConfig
    
    # Create task instance
    task = ChartQAPro(TaskConfig(max_dataset_len=10))
    
    # Load dataset
    print("Loading ChartQAPro dataset...")
    ds = task.dataset
    print(f"Dataset size: {len(ds)}")
    
    # Test with first example
    example = ds[0]
    print(f"\nFirst example:")
    print(f"  ID: {task.doc_to_id(example)}")
    print(f"  Question: {task.doc_to_text(example)}")
    print(f"  Question Type: {example.get('Question Type', 'N/A')}")
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