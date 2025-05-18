from .heron_bench_scorer import HeronBenchScorer
from .exact_match_scorer import ExactMatchScorer
from .llm_as_a_judge_scorer import LlmAsaJudgeScorer
from .rougel_scorer import RougeLScorer
from .substring_match_scorer import SubstringMatchScorer
from .scorer import Scorer
from .jmmmu_scorer import JMMMUScorer
from .mmmu_scorer import MMMUScorer
from .jdocqa_scorer import JDocQAScorer
from .jic_vqa_scorer import JICVQAScorer
from .mecha_ja_scorer import MECHAJaScorer
from .cc_ocr_scorer import CCOCRScorer
from .scorer import ScorerConfig
from typing import Callable


class ScorerRegistry:
    """Registry to map metrics to their corresponding scorer classes."""

    _scorers: dict[str, Callable[[ScorerConfig], Scorer]] = {
        "heron-bench": HeronBenchScorer,
        "exact-match": ExactMatchScorer,
        "llm-as-a-judge": LlmAsaJudgeScorer,
        "rougel": RougeLScorer,
        "substring-match": SubstringMatchScorer,
        "jmmmu": JMMMUScorer,
        "jdocqa": JDocQAScorer,
        "mmmu": MMMUScorer,
        "jic-vqa": JICVQAScorer,
        "mecha-ja": MECHAJaScorer,
        "cc-ocr": CCOCRScorer,
    }

    @classmethod
    def get_metric_list(cls) -> list[str]:
        """Get a list of supported metrics."""
        return list(cls._scorers.keys())

    @classmethod
    def load_scorer(
        cls, metric: str, scorer_config: ScorerConfig = ScorerConfig()
    ) -> Scorer:
        """Load a scorer instance from the scorer registry."""
        try:
            return cls._scorers[metric](scorer_config)  # type: ignore
        except KeyError:
            raise ValueError(f"Metric '{metric}' is not supported.")
