from eval_mm.tasks.task import Task
from eval_mm.tasks.task_registry import register_task
from datasets import load_dataset, Dataset
from PIL import Image


@register_task("textvqa")
class TextVQA(Task):
    """TextVQA task implementation.
    
    TextVQA requires models to read and reason about text in images to answer questions.
    It tests the model's ability to incorporate text present in images and reason over it.
    """
    
    def __init__(self, config):
        super().__init__(config)
    
    def _prepare_dataset(self) -> Dataset:
        """Load TextVQA validation set."""
        # Load the TextVQA dataset from lmms-lab
        ds = load_dataset("lmms-lab/textvqa", split="validation")
        
        return ds

    def _prepare_test_dataset(self) -> Dataset:
        # Stream a tiny subset to avoid heavy downloads/cache writes in CI
        n = getattr(self.config, "max_dataset_len", 10)
        stream = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)
        buf = {
            "question_id": [],
            "question": [],
            "answers": [],
            "image": [],
        }
        count = 0
        for ex in stream:
            buf["question_id"].append(str(ex["question_id"]))
            buf["question"].append(ex["question"])
            buf["answers"].append(ex["answers"])  # list[str]
            buf["image"].append(ex["image"])      # keep image column for lazy decode
            count += 1
            if count >= n:
                break
        return Dataset.from_dict(buf)
    
    @staticmethod
    def doc_to_text(doc) -> str:
        """Convert document to text prompt.
        
        TextVQA is an extractive QA task, so we just return the question.
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
        
        TextVQA provides multiple valid answers for each question.
        We return all of them for evaluation with substring-match scorer.
        """
        return doc['answers']


def test_textvqa_task():
    """Basic loader/type checks for TextVQA."""
    from eval_mm.tasks.task import TaskConfig

    task = TextVQA(TaskConfig(max_dataset_len=10))
    ds = task.dataset
    assert len(ds) <= 10
    ex = ds[0]
    # Verify data shapes/types without verbose prints
    assert isinstance(task.doc_to_text(ex), str)
    vis = task.doc_to_visual(ex)
    assert isinstance(vis, list) and isinstance(vis[0], Image.Image)
    assert isinstance(task.doc_to_id(ex), str)
    answers = task.doc_to_answer(ex)
    assert isinstance(answers, list) and all(isinstance(a, str) for a in answers)
