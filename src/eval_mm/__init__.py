import os as _os
import warnings as _warnings

from dotenv import load_dotenv as _load_dotenv
from .tasks.task_registry import TaskRegistry
from .tasks.task import TaskConfig
from .metrics.scorer_registry import ScorerRegistry
from .metrics.scorer import ScorerConfig
from .utils.azure_client import OpenAIChatAPI
from .models.generation_config import GenerationConfig
from .models.base_vlm import BaseVLM
from .runner import run_evaluation
from .result_schema import RunManifest, load_manifest, load_predictions, load_evaluation

# Load environment variables
_load_dotenv()

# Guard: HuggingFace cache must NOT be under $HOME
_hf_home = _os.environ.get("HF_HOME", "")
_home = _os.path.expanduser("~")
if not _hf_home or _hf_home.startswith(_home):
    _warnings.warn(
        f"HF_HOME is {'not set' if not _hf_home else 'under $HOME (' + _hf_home + ')'}.  "
        "Set HF_HOME to a shared path (e.g. data/shared/models/huggingface) "
        "to avoid filling the home directory with large model caches.",
        stacklevel=1,
    )

__all__ = [
    "BaseVLM",
    "GenerationConfig",
    "OpenAIChatAPI",
    "ScorerConfig",
    "ScorerRegistry",
    "TaskConfig",
    "TaskRegistry",
    "run_evaluation",
    "RunManifest",
    "load_manifest",
    "load_predictions",
    "load_evaluation",
]
