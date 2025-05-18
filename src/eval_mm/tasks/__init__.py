from .ja_vg_vqa_500 import JaVGVQA500
from .japanese_heron_bench import JapaneseHeronBench
from .ja_vlm_bench_in_the_wild import JaVLMBenchIntheWild
from .jmmmu import JMMMU
from .ja_multi_image_vqa import JAMultiImageVQA
from .jdocqa import JDocQA
from .mmmu import MMMU
from .llava_bench_in_the_wild import LlavaBenchIntheWild
from .jic_vqa import JICVQA
from .mecha_ja import MECHAJa
from .mmmlu import MMMLU
from .cc_ocr import CCOCR
from .cvqa import CVQA
from .task_registry import TaskRegistry
from .task import TaskConfig

__all__ = [
    "JaVGVQA500",
    "JapaneseHeronBench",
    "JaVLMBenchIntheWild",
    "JMMMU",
    "JAMultiImageVQA",
    "JDocQA",
    "MMMU",
    "LlavaBenchIntheWild",
    "JICVQA",
    "MECHAJa",
    "MMMLU",
    "CCOCR",
    "CVQA",
    "TaskRegistry",
    "TaskConfig",
]
