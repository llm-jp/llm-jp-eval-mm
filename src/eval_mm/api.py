"""Lightweight FastAPI backend for the eval_mm web frontend.

Run with:
    uvicorn eval_mm.api:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="eval_mm API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/tasks")
def list_tasks() -> list[dict]:
    from eval_mm.metadata import TASKS

    return [
        {
            "task_id": t.task_id,
            "display_name": t.display_name,
            "cluster": t.cluster,
        }
        for t in TASKS.values()
    ]


@app.get("/api/models")
def list_models() -> list[str]:
    from eval_mm.metadata import LEADERBOARD_MODELS

    return LEADERBOARD_MODELS


@app.get("/api/metrics")
def list_metrics() -> list[dict]:
    from eval_mm.metadata import METRICS

    return [
        {
            "metric_id": m.metric_id,
            "display_name": m.display_name,
        }
        for m in METRICS.values()
    ]
