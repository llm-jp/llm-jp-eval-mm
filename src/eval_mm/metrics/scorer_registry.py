"""
Scorer registry with decorator-based registration to avoid duplication.
"""

from typing import Type, Callable
from .scorer import Scorer, ScorerConfig

# Global registry dictionary
_scorer_registry: dict[str, Type[Scorer]] = {}


def register_scorer(*names: str):
    """
    Decorator to register a scorer class in the global registry.
    Can register multiple names for the same scorer.
    
    Usage:
        @register_scorer("my-scorer-name", "MyScorer", "MY_SCORER")
        class MyScorer(Scorer):
            ...
    """
    def decorator(cls: Type[Scorer]) -> Type[Scorer]:
        for name in names:
            _scorer_registry[name] = cls
        return cls
    return decorator


class ScorerRegistry:
    """Registry to map metrics to their corresponding scorer classes."""

    @classmethod
    def get_metric_list(cls) -> list[str]:
        """Get a list of supported metrics."""
        return list(_scorer_registry.keys())

    @classmethod
    def load_scorer(
        cls, metric: str, scorer_config: ScorerConfig = ScorerConfig()
    ) -> Scorer:
        """Load a scorer instance from the scorer registry."""
        try:
            return _scorer_registry[metric](scorer_config)
        except KeyError:
            raise ValueError(f"Metric '{metric}' is not supported.")