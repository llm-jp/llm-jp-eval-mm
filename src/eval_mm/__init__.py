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
