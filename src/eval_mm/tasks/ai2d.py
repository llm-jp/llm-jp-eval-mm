from eval_mm.tasks.task import Task
from eval_mm.tasks.task_registry import register_task
from datasets import load_dataset, Dataset
from PIL import Image


@register_task("ai2d")
class AI2D(Task):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset("lmms-lab/ai2d", split="test")
        ds = ds.map(lambda example, idx: {"question_id": idx}, with_indices=True)
        return ds

    @staticmethod
    def doc_to_text(doc) -> str:
        question = doc["question"]
        choices = doc["options"]
        len_choices = len(choices)
        
        pre_prompt = ""
        post_prompt = "\nAnswer with the option's letter from the given choices directly."
        
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        
        return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        return [doc['image']]

    @staticmethod
    def doc_to_id(doc) -> str:
        return str(doc['question_id'])

    @staticmethod
    def doc_to_answer(doc) -> str:
        answer_idx = int(doc['answer'])
        return chr(ord('A') + answer_idx)


def test_task():
    from eval_mm.tasks.task import TaskConfig

    task = AI2D(TaskConfig())
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)

if __name__ == "__main__":
    test_task()
