from datasets import Dataset, load_dataset
from .task import Task
from .task_registry import register_task
from PIL import Image
import os

MULTI_CHOICE_PROMPT = (
    "与えられた選択肢の中から最も適切な回答のアルファベットを直接記入してください。"
)

OPTIONS_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join(
        [
            f"{option_letter}. {option}"
            for option_letter, option in zip(option_letters, options)
        ]
    )
    return choices_str


def construct_prompt(question, options):
    parsed_options = parse_options(options)
    return f"{question}\n{parsed_options}\n\n{MULTI_CHOICE_PROMPT}"


@register_task("cvqa")
class CVQA(Task):
    default_metric = "substring-match"

    def _prepare_dataset(self) -> Dataset:
        ds = load_dataset("afaji/cvqa", split="test")
        ds = ds.filter(lambda x: x["Subset"] == "('Japanese', 'Japan')")
        ds = ds.map(
            lambda x, idx: {
                "index": str(idx),
                "question_id": str(idx),
                "question": x["Question"],
                "question_en": x.get("Translated Question"),  # English (optional)
                "options": x["Options"],
                "translated_options": x.get("Translated Options"),  # English (optional)
                "answer": x["Label"],  # 0~3
                "answer_text": OPTIONS_MAP[x["Label"]],
                # keep original `image` column as-is for lazy decode
            },
            with_indices=True,
        )
        return ds

    def _prepare_test_dataset(self) -> Dataset:
        # Stream to pick the first N Japanese samples and build a tiny Dataset
        n = getattr(self.config, "max_dataset_len", 10)
        stream = load_dataset("afaji/cvqa", split="test", streaming=True)
        buf = {
            "index": [],
            "question_id": [],
            "question": [],
            "question_en": [],
            "options": [],
            "translated_options": [],
            "answer": [],
            "answer_text": [],
            "image": [],
        }
        count = 0
        for ex in stream:
            if ex.get("Subset") == "('Japanese', 'Japan')":
                buf["index"].append(str(count))
                buf["question_id"].append(str(count))
                buf["question"].append(ex["Question"])
                buf["question_en"].append(ex.get("Translated Question"))
                buf["options"].append(ex["Options"])
                buf["translated_options"].append(ex.get("Translated Options"))
                buf["answer"].append(ex["Label"])  # 0~3
                buf["answer_text"].append(OPTIONS_MAP[ex["Label"]])
                buf["image"].append(ex["image"])  # keep original to lazily decode later
                count += 1
                if count >= n:
                    break
        return Dataset.from_dict(buf)

    @staticmethod
    def doc_to_text(doc) -> str:
        # Lazily construct the prompt to reduce preprocessing cost
        return construct_prompt(doc["question"], doc["options"])

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        return [doc["image"]]

    @staticmethod
    def doc_to_id(doc) -> str:
        return str(doc["question_id"])

    @staticmethod
    def doc_to_answer(doc) -> str:
        return doc["answer_text"]


def test_task():
    from eval_mm.tasks.task import TaskConfig

    task = CVQA(TaskConfig(max_dataset_len=10))
    ds = task.dataset
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
    # Intentionally avoid printing entire example to prevent accidental image decode


if __name__ == "__main__":
    test_task()
