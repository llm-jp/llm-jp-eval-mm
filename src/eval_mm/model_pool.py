"""The model pool: which models the leaderboard evaluates and how to serve them."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

POOL_PATH = Path(__file__).with_name("model_pool.toml")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str = ""
    params_b: float | None = None
    source: str = "vllm"
    status: str = "active"
    gpus: int = 1
    thinking: str | None = None
    serve_args: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        if self.source not in ("vllm", "api"):
            raise ValueError(f"{self.id}: unknown source {self.source!r}")
        if self.status not in ("active", "pending"):
            raise ValueError(f"{self.id}: unknown status {self.status!r}")
        if self.thinking not in (None, "on", "off"):
            raise ValueError(f"{self.id}: unknown thinking {self.thinking!r}")


def load_pool(path: Path = POOL_PATH) -> list[ModelSpec]:
    with open(path, "rb") as f:
        specs = [ModelSpec(**entry) for entry in tomllib.load(f)["models"]]
    ids = [s.id for s in specs]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(f"Duplicated model IDs in {path}: {duplicated}")
    return specs


def get_spec(model_id: str, path: Path = POOL_PATH) -> ModelSpec:
    for spec in load_pool(path):
        if spec.id == model_id:
            return spec
    raise KeyError(f"{model_id} is not in the model pool ({path})")
