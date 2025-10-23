from examples.model_registry_data import (
    DEFAULT_API_ENV_GROUP,
    DEFAULT_TRANSFORMERS_ENV_GROUP,
)
from examples.model_table import get_model_spec, get_supported_model_ids
from pathlib import Path
import sys

try:
    import pytest
except ModuleNotFoundError:
    from contextlib import contextmanager

    @contextmanager
    def _raises(expected_exception):
        try:
            yield
        except expected_exception:
            return
        raise AssertionError(
            f"expected exception {expected_exception.__name__} not raised"
        )

    class _PytestStub:
        raises = staticmethod(_raises)

    pytest = _PytestStub()

sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_pixtral_defaults_to_vllm_runtime():
    spec = get_model_spec("mistralai/Pixtral-12B-2409")
    assert spec.default_runtime == "vllm"
    cfg = spec.get_runtime_config("vllm")
    assert cfg.module_path == "examples.runtimes.vllm.pixtral.VLM"
    assert cfg.env_group == "pixtral"


def test_internvl3_supports_both_runtimes():
    spec = get_model_spec("OpenGVLab/InternVL3-8B")
    assert spec.default_runtime == "transformers"
    assert spec.has_runtime("vllm")
    vllm_cfg = spec.get_runtime_config("vllm")
    assert vllm_cfg.module_path == "examples.runtimes.vllm.base.VLLM"


def test_get_runtime_config_raises_for_missing_runtime():
    spec = get_model_spec("llava-hf/llava-1.5-7b-hf")
    with pytest.raises(ValueError):
        spec.get_runtime_config("vllm")


def test_filter_models_by_runtime():
    vllm_models = get_supported_model_ids(runtime="vllm")
    assert "mistralai/Pixtral-12B-2409" in vllm_models
    assert "llava-hf/llava-1.5-7b-hf" not in vllm_models


def test_api_runtime_present_for_gpt4o():
    spec = get_model_spec("gpt-4o-2024-11-20")
    assert spec.default_runtime == "api"
    api_cfg = spec.get_runtime_config()
    assert api_cfg.module_path == "examples.runtimes.api.gpt4o.VLM"
    assert api_cfg.env_group == DEFAULT_API_ENV_GROUP


def test_transformer_env_group_override():
    cfg = get_model_spec("cyberagent/llava-calm2-siglip").get_runtime_config()
    assert cfg.env_group == "calm"


def test_transformer_env_group_default():
    cfg = get_model_spec("llava-hf/llava-1.5-7b-hf").get_runtime_config()
    assert cfg.env_group == DEFAULT_TRANSFORMERS_ENV_GROUP
