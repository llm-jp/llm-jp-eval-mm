"""Backward-compatible re-export. Use ``from eval_mm import GenerationConfig`` instead."""

import warnings

warnings.warn(
    "Importing from examples/utils.py is deprecated. "
    "Use ``from eval_mm import GenerationConfig`` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from eval_mm.models.generation_config import GenerationConfig  # noqa: F401
