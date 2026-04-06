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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


SCHEMA_VERSION = "1.0"


@dataclass
class RunManifest:
    """Metadata written alongside result files."""

    schema_version: str = SCHEMA_VERSION
    model_id: str = ""
    task_id: str = ""
    metrics: list[str] = field(default_factory=list)
    created_at: str = ""
    result_dir: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        return cls(**json.loads(text))

    @classmethod
    def from_file(cls, path: str) -> RunManifest:
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())


# ── IO helpers ─────────────────────────────────────────────────────

def write_manifest(
    output_dir: str,
    model_id: str,
    task_id: str,
    metrics: list[str],
) -> str:
    """Write manifest.json and return its path."""
    manifest = RunManifest(
        model_id=model_id,
        task_id=task_id,
        metrics=metrics,
        created_at=datetime.now(timezone.utc).isoformat(),
        result_dir=output_dir,
    )
    path = os.path.join(output_dir, "manifest.json")
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


