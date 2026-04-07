"""Lightweight FastAPI backend for the eval_mm web frontend.

Run with:
    uvicorn eval_mm.api:app --reload
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="eval_mm API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root directory where evaluation results are stored.
# Override via the EVAL_MM_RESULT_DIR environment variable.
RESULT_DIR = os.environ.get("EVAL_MM_RESULT_DIR", "result")


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


# ── Result browsing endpoints ────────────────────────────────────


def _discover_results(result_dir: str) -> list[dict]:
    """Walk the result directory and return manifest info for each run."""
    from eval_mm.result_schema import RunManifest

    results: list[dict] = []
    if not os.path.isdir(result_dir):
        return results

    for task_id in sorted(os.listdir(result_dir)):
        task_path = os.path.join(result_dir, task_id)
        if not os.path.isdir(task_path):
            continue
        for model_id in sorted(os.listdir(task_path)):
            model_path = os.path.join(task_path, model_id)
            if not os.path.isdir(model_path):
                continue
            manifest_path = os.path.join(model_path, "manifest.json")
            if os.path.isfile(manifest_path):
                try:
                    m = RunManifest.from_file(manifest_path)
                    results.append(
                        {
                            "task_id": m.task_id or task_id,
                            "model_id": m.model_id or model_id,
                            "metrics": m.metrics,
                            "created_at": m.created_at,
                        }
                    )
                except Exception:
                    # Skip malformed manifests
                    results.append(
                        {
                            "task_id": task_id,
                            "model_id": model_id,
                            "metrics": [],
                            "created_at": "",
                        }
                    )
            else:
                # Directory exists but no manifest — still expose it
                has_predictions = os.path.isfile(
                    os.path.join(model_path, "prediction.jsonl")
                )
                has_evaluation = os.path.isfile(
                    os.path.join(model_path, "evaluation.jsonl")
                )
                if has_predictions or has_evaluation:
                    results.append(
                        {
                            "task_id": task_id,
                            "model_id": model_id,
                            "metrics": [],
                            "created_at": "",
                        }
                    )
    return results


@app.get("/api/results")
def list_results(
    result_dir: str = Query(default=None, description="Override result directory"),
) -> list[dict]:
    """List available evaluation results across all task/model combinations."""
    directory = result_dir or RESULT_DIR
    return _discover_results(directory)


@app.get("/api/predictions/{task_id}/{model_id:path}")
def get_predictions(
    task_id: str,
    model_id: str,
    result_dir: str = Query(default=None, description="Override result directory"),
    offset: int = Query(default=0, ge=0, description="Skip first N predictions"),
    limit: int = Query(default=100, ge=1, le=10000, description="Max predictions to return"),
) -> dict:
    """Get predictions for a specific task/model combination.

    Returns paginated predictions from prediction.jsonl.
    The *model_id* may contain slashes (e.g. ``Qwen/Qwen2-VL-7B-Instruct``).
    """
    from eval_mm.result_schema import load_predictions

    directory = result_dir or RESULT_DIR
    output_dir = os.path.join(directory, task_id, model_id)

    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail=f"No results for {task_id}/{model_id}")

    prediction_path = os.path.join(output_dir, "prediction.jsonl")
    if not os.path.isfile(prediction_path):
        raise HTTPException(status_code=404, detail="prediction.jsonl not found")

    preds = load_predictions(output_dir)
    total = len(preds)
    page = preds[offset : offset + limit]

    return {
        "task_id": task_id,
        "model_id": model_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "predictions": page,
    }


@app.get("/api/scores/{task_id}")
def get_scores(
    task_id: str,
    result_dir: str = Query(default=None, description="Override result directory"),
) -> dict:
    """Get aggregate scores for a task across all models.

    Reads evaluation.jsonl from each model sub-directory under the given task.
    """
    from eval_mm.result_schema import load_evaluation

    directory = result_dir or RESULT_DIR
    task_path = os.path.join(directory, task_id)

    if not os.path.isdir(task_path):
        raise HTTPException(status_code=404, detail=f"No results for task {task_id}")

    scores: list[dict] = []
    for model_id in sorted(os.listdir(task_path)):
        model_path = os.path.join(task_path, model_id)
        if not os.path.isdir(model_path):
            continue
        eval_path = os.path.join(model_path, "evaluation.jsonl")
        if not os.path.isfile(eval_path):
            continue
        try:
            metrics = load_evaluation(model_path)
            scores.append({"model_id": model_id, "metrics": metrics})
        except Exception:
            # Skip unreadable evaluation files
            continue

    return {"task_id": task_id, "models": scores}
