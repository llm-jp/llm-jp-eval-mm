from datasets import Dataset, load_dataset
from .task import Task
from PIL import Image

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


class CVQA(Task):
    default_metric = "substring-match"

    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset("afaji/cvqa", split="test")

        ds = ds.filter(lambda x: x["Subset"] == "('Japanese', 'Japan')")

        ds = ds.map(
            lambda x, idx: {
                "index": str(idx),
                "question_id": str(idx),
                "question": x["Question"],
                "question_en": x["Translated Question"],  # English
                "options": x["Options"],
                "translated_options": x["Translated Options"],  # English
                "answer": x["Label"],  # 0~3
                "answer_text": OPTIONS_MAP[x["Label"]],
                "input_text": construct_prompt(x["Question"], x["Options"]),
                "image": x["image"],
            },
            with_indices=True,
        )

        return ds

    @staticmethod
    def doc_to_text(doc) -> str:
        return doc["input_text"]

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

    task = CVQA(TaskConfig())
    ds = task.dataset
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), str)
    assert isinstance(task.doc_to_answer(ds[0]), str)
    print(ds[0])


if __name__ == "__main__":
    test_task()
