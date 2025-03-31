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
from .mnist import MNIST
from .task import TaskConfig, Task


class TaskRegistry:
    """Registry to map metrics to their corresponding scorer classes."""

    _tasks = {
        "japanese-heron-bench": JapaneseHeronBench,
        "ja-vlm-bench-in-the-wild": JaVLMBenchIntheWild,
        "ja-vg-vqa-500": JaVGVQA500,
        "jmmmu": JMMMU,
        "ja-multi-image-vqa": JAMultiImageVQA,
        "jdocqa": JDocQA,
        "mmmu": MMMU,
        "llava-bench-in-the-wild": LlavaBenchIntheWild,
        "jic-vqa": JICVQA,
        "mecha-ja": MECHAJa,
        "mmmlu": MMMLU,
        "mnist": MNIST,
    }

    @classmethod
    def get_task_list(cls):
        """Get the list of supported tasks."""
        return list(cls._tasks.keys())

    @classmethod
    def load_task(cls, task_name: str, task_config: TaskConfig = TaskConfig()) -> Task:
        """Load a task instance from the task registry."""
        try:
            return cls._tasks[task_name](task_config)
        except KeyError:
            raise ValueError(f"Task '{task_name}' is not supported.")
