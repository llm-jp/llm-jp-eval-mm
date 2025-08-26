from eval_mm.tasks.task import Task
from eval_mm.tasks.task_registry import register_task
from datasets import load_dataset, Dataset, concatenate_datasets
from PIL import Image


@register_task("blink")
class BLINK(Task):
    """BLINK Benchmark task implementation.
    
    BLINK is a multimodal benchmark with 14 different visual perception tasks.
    All tasks are formulated as multiple-choice questions.
    """
    
    # All available BLINK configs
    CONFIGS = [
        'Art_Style', 'Counting', 'Forensic_Detection', 'Functional_Correspondence',
        'IQ_Test', 'Jigsaw', 'Multi-view_Reasoning', 'Object_Localization',
        'Relative_Depth', 'Relative_Reflectance', 'Semantic_Correspondence',
        'Spatial_Relation', 'Visual_Correspondence', 'Visual_Similarity'
    ]
    
    def __init__(self, config):
        super().__init__(config)
    
    @staticmethod
    def _prepare_dataset() -> Dataset:
        """Load and concatenate all BLINK configs into a single dataset."""
        all_datasets = []
        
        for config_name in BLINK.CONFIGS:
            # Load validation split for each config
            ds = load_dataset("BLINK-Benchmark/BLINK", config_name, split="val")
            # Add config name to each example for tracking
            ds = ds.map(lambda x: {"config_name": config_name})
            all_datasets.append(ds)
        
        # Concatenate all configs
        combined_dataset = concatenate_datasets(all_datasets)
        
        # Add unique question_id
        combined_dataset = combined_dataset.map(
            lambda example, idx: {"question_id": f"blink_val_{idx}"}, 
            with_indices=True
        )
        
        return combined_dataset
    
    @staticmethod
    def doc_to_text(doc) -> str:
        """Convert document to text prompt.
        
        BLINK already provides a formatted 'prompt' field that includes
        the question and choices in the correct format.
        """

        pre_prompt = ""
        post_prompt = "\nAnswer with the option's letter from the given choices directly."

        return f"{pre_prompt}{doc['prompt']}{post_prompt}"

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        """Extract images from document.
        
        BLINK can have 1-4 images depending on the task.
        We collect all non-None images.
        """
        images = []
        for i in range(1, 5):  # image_1 through image_4
            img_key = f'image_{i}'
            if img_key in doc and doc[img_key] is not None:
                images.append(doc[img_key])
        return images
    
    @staticmethod
    def doc_to_id(doc) -> str:
        """Return unique question ID."""
        return doc['question_id']
    
    @staticmethod
    def doc_to_answer(doc) -> str:
        """Extract answer letter from the format '(A)' -> 'A'."""
        answer = doc['answer']
        # Remove parentheses if present
        if answer.startswith('(') and answer.endswith(')'):
            return answer[1:-1]
        return answer


def test_blink_task():
    """Test BLINK task implementation."""
    from eval_mm.tasks.task import TaskConfig
    
    # Create task instance
    task = BLINK(TaskConfig())
    
    # Load dataset
    print("Loading BLINK dataset...")
    ds = task.dataset
    print(f"Total examples: {len(ds)}")
    
    # Test with first example
    example = ds[0]
    print(f"\nFirst example:")
    print(f"  Config: {example['config_name']}")
    print(f"  ID: {task.doc_to_id(example)}")
    print(f"  Question: {example['question']}")
    print(f"  Num images: {len(task.doc_to_visual(example))}")
    print(f"  Num choices: {len(example['choices'])}")
    print(f"  Answer: {task.doc_to_answer(example)}")
    print(f"  Text prompt preview: {task.doc_to_text(example)[:200]}...")
    
    # Verify data types
    assert isinstance(task.doc_to_text(example), str)
    assert isinstance(task.doc_to_visual(example), list)
    assert all(isinstance(img, Image.Image) for img in task.doc_to_visual(example))
    assert isinstance(task.doc_to_id(example), str)
    assert isinstance(task.doc_to_answer(example), str)
    assert task.doc_to_answer(example) in ['A', 'B', 'C', 'D']
    
    print("\nAll tests passed!")
