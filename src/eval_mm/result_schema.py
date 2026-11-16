"""Result schema — canonical format for evaluation artifacts.

All consumers (CLI summary, Streamlit browser, leaderboard generator,
GitHub Pages) should read/write through this module instead of doing
ad-hoc JSONL parsing.

Directory layout per (task, model):
    result/<task_id>/<model_id>/
        manifest.json       -- metadata about the run
        prediction.jsonl     -- per-sample predictions + scores
        evaluation.jsonl     -- aggregated metrics
        error_message.jsonl  -- (optional) inference errors
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata


SCHEMA_VERSION = "1.1"


@dataclass
class RunManifest:
    """Metadata written alongside result files."""

    schema_version: str = SCHEMA_VERSION
    model_id: str = ""
    task_id: str = ""
    metrics: list[str] = field(default_factory=list)
    created_at: str = ""
    result_dir: str = ""
    # Added in 1.1: what is needed to reproduce the run.
    judge_model: str = ""
    generation: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in json.loads(text).items() if k in known})

    @classmethod
    def from_file(cls, path: str) -> RunManifest:
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())


# ── IO helpers ─────────────────────────────────────────────────────

_RECORDED_PACKAGES = ("eval-mm", "datasets", "openai", "vllm", "transformers", "torch")


def collect_environment() -> dict:
    """Versions that can change scores: this repository's commit and key packages."""
    packages = {}
    for name in _RECORDED_PACKAGES:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return {"git_commit": commit, "packages": packages}


def write_manifest(
    output_dir: str,
    model_id: str,
    task_id: str,
    metrics: list[str],
    *,
    judge_model: str = "",
    generation: dict | None = None,
    dataset: dict | None = None,
) -> str:
    """Write manifest.json and return its path.

    Scoring often runs separately from inference. When *generation* or
    *dataset* is ``None``, the values recorded at inference time are kept.
    """
    path = os.path.join(output_dir, "manifest.json")
    previous = RunManifest.from_file(path) if os.path.exists(path) else RunManifest()
    manifest = RunManifest(
        model_id=model_id,
        task_id=task_id,
        metrics=metrics,
        created_at=datetime.now(timezone.utc).isoformat(),
        result_dir=output_dir,
        judge_model=judge_model,
        generation=previous.generation if generation is None else generation,
        dataset=previous.dataset if dataset is None else dataset,
        environment=collect_environment(),
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(manifest.to_json())
    return path


def load_manifest(output_dir: str) -> RunManifest:
    """Load manifest.json from a result directory."""
    return RunManifest.from_file(os.path.join(output_dir, "manifest.json"))


def load_predictions(output_dir: str) -> list[dict]:
    """Load prediction.jsonl from a result directory."""
    path = os.path.join(output_dir, "prediction.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_evaluation(output_dir: str) -> dict:
    """Load evaluation.jsonl (aggregated metrics) from a result directory."""
    path = os.path.join(output_dir, "evaluation.jsonl")
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())


