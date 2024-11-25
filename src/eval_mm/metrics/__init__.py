from .heron_bench_scorer import HeronBenchScorer
from .exact_match_scorer import ExactMatchScorer
from .llm_as_a_judge_scorer import LlmAsaJudgeScorer
from .rougel_scorer import RougeLScorer
from .substring_match_scorer import SubstringMatchScorer
from .scorer import Scorer
from .jmmmu_scorer import JMMMUScorer
from .jdocqa_scorer import JDocQAScorer


class ScorerRegistry:
    """Registry to map metrics to their corresponding scorer classes."""

    _scorers = {
        "llm_as_a_judge_heron_bench": HeronBenchScorer,
        "exact_match": ExactMatchScorer,
        "llm_as_a_judge": LlmAsaJudgeScorer,
        "rougel": RougeLScorer,
        "substring_match": SubstringMatchScorer,
        "jmmmu": JMMMUScorer,
        "jdocqa": JDocQAScorer,
    }

    @classmethod
    def register(cls, metric: str, scorer_class: type):
        """Register a new scorer for a metric."""
        cls._scorers[metric] = scorer_class

    @classmethod
    def get_scorer(cls, metric: str) -> Scorer:
        """Get the scorer class for the given metric."""
        try:
            return cls._scorers[metric]
        except KeyError:
            raise ValueError(f"Metric '{metric}' is not supported.")
