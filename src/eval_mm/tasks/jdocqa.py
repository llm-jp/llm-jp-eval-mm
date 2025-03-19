from datasets import Dataset, load_dataset
from pdf2image import convert_from_path

from ..api.registry import register_task
from ..api.task import Task

import aiohttp
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images


def get_elements_from_index(indices_str, array):
    try:
        indices = [int(x.strip()) - 1 for x in indices_str.split(",")]
        elements = [array[i] for i in indices if 0 <= i < len(array)]
        return elements
    except ValueError:
        print("The string doesn't seem to have numbers or commas in the right places.")
        return None  # Or maybe an empty list, depending on how you wanna handle it
    except IndexError:
        print("Out of bounds error!")
        return None  # Same, an empty list or special value could work


@register_task("jdocqa")
class JDocQA(Task):
    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset(
            "shunk031/JDocQA",
            split="test",
            rename_pdf_category=True,
            trust_remote_code=True,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
            },
        )
        ds = ds.rename_column("question", "input_text")
        ds = ds.map(lambda example, idx: {"question_id": idx}, with_indices=True)
        keep_columns = [
            "input_text",
            "pdf_filepath",
            "question_page_number",
            "question_id",
            "answer",
            "answer_type",
        ]
        ds = ds.remove_columns(
            [col for col in ds.column_names if col not in keep_columns]
        )
        return ds

    @staticmethod
    def doc_to_text(doc) -> str:
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc) -> list[Image.Image]:
        images_all = pdf_to_images(doc["pdf_filepath"])
        images = get_elements_from_index(doc["question_page_number"], images_all)
        return images

    @staticmethod
    def doc_to_id(doc) -> int:
        return doc["question_id"]

    @staticmethod
    def doc_to_answer(doc) -> str:
        return doc["answer"]


def test_task():
    from eval_mm.api.task import TaskConfig

    task = JDocQA(TaskConfig())
    ds = task.dataset
    print(ds[0])
    assert isinstance(task.doc_to_text(ds[0]), str)
    assert isinstance(task.doc_to_visual(ds[0]), list)
    assert isinstance(task.doc_to_visual(ds[0])[0], Image.Image)
    assert isinstance(task.doc_to_id(ds[0]), int)
    assert isinstance(task.doc_to_answer(ds[0]), str)
