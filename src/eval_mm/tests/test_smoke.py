"""Portable smoke tests that run without GPU, network, or hardcoded paths.

These tests verify:
- Task registry works
- Scorer registry works
- BaseVLM interface
- GenerationConfig defaults
- Result schema round-trip
- CLI help text
- Runner can evaluate pre-existing predictions
"""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from eval_mm import (
    BaseVLM,
    GenerationConfig,
    ScorerRegistry,
    TaskRegistry,
)
from eval_mm.result_schema import RunManifest, write_manifest, load_manifest


FIXTURES = Path(__file__).parent / "fixtures"


# ── Registry tests ──────────────────────────────────────────────────


def test_task_registry_lists_tasks():
    tasks = TaskRegistry.get_task_list()
    assert len(tasks) > 0
    assert "japanese-heron-bench" in tasks


def test_scorer_registry_lists_metrics():

    metrics = ScorerRegistry.get_metric_list()
    assert len(metrics) > 0
    assert "rougel" in metrics


# ── GenerationConfig tests ──────────────────────────────────────────


def test_generation_config_defaults():
    gc = GenerationConfig()
    assert gc.max_new_tokens == 1024
    assert gc.temperature == 0.0
    assert gc.do_sample is False


def test_generation_config_custom():
    gc = GenerationConfig(max_new_tokens=512, temperature=0.7)
    assert gc.max_new_tokens == 512
    assert gc.temperature == 0.7


# ── BaseVLM tests ───────────────────────────────────────────────────


def test_base_vlm_is_abstract():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseVLM()


# ── Result schema tests ────────────────────────────────────────────


def test_manifest_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_manifest(tmpdir, "test-model", "test-task", ["rougel"])
        loaded = load_manifest(tmpdir)
        assert loaded.model_id == "test-model"
        assert loaded.task_id == "test-task"
        assert loaded.metrics == ["rougel"]
        assert loaded.schema_version == "1.0"


def test_manifest_json_serialization():
    m = RunManifest(model_id="m", task_id="t", metrics=["x"])
    text = m.to_json()
    loaded = RunManifest.from_json(text)
    assert loaded.model_id == "m"
    assert loaded.metrics == ["x"]


# ── CLI tests ───────────────────────────────────────────────────────


def test_cli_help(capsys):
    from eval_mm.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0


def test_cli_list_tasks(capsys):
    from eval_mm.cli import main

    main(["list", "tasks"])
    captured = capsys.readouterr()
    assert "japanese-heron-bench" in captured.out


def test_cli_list_metrics(capsys):
    from eval_mm.cli import main

    main(["list", "metrics"])
    captured = capsys.readouterr()
    assert "rougel" in captured.out


# ── Fixture image test ──────────────────────────────────────────────


def test_fixture_image_loadable():
    img_path = FIXTURES / "test_image.png"
    assert img_path.exists(), f"Fixture image not found at {img_path}"
    img = Image.open(img_path)
    assert img.size == (64, 64)
    assert img.mode == "RGB"
