"""Lightweight FastAPI backend for the eval_mm web frontend.

Run with:
    uvicorn eval_mm.api:app --reload
"""

from __future__ import annotations

import json
import os
import subprocess

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
_default_result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "result")
RESULT_DIR = os.environ.get("EVAL_MM_RESULT_DIR", os.path.normpath(_default_result_dir))


# ── Helpers ────────────────────────────────────────────────────


def _has_result_files(path: str) -> bool:
    """Return True if *path* contains any known result artefact."""
    return any(
        os.path.isfile(os.path.join(path, f))
        for f in ("prediction.jsonl", "evaluation.jsonl", "manifest.json", "error_message.jsonl")
    )


def _iter_model_dirs(task_path: str):
    """Yield ``(model_id, model_path)`` under a task directory.

    Handles both flat (``model/``) and nested (``org/model/``) layouts.
    """
    for entry in sorted(os.listdir(task_path)):
        entry_path = os.path.join(task_path, entry)
        if not os.path.isdir(entry_path):
            continue
        if _has_result_files(entry_path):
            yield entry, entry_path
        else:
            for sub in sorted(os.listdir(entry_path)):
                sub_path = os.path.join(entry_path, sub)
                if os.path.isdir(sub_path) and _has_result_files(sub_path):
                    yield f"{entry}/{sub}", sub_path


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ── GPU monitoring ──────────────────────────────────────────────


@app.get("/api/gpus")
def get_gpus() -> list[dict]:
    """Return real-time GPU stats from nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return []

    if result.returncode != 0:
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, util, mem_used, mem_total, temp = parts
        utilization = int(util)
        if utilization >= 80:
            status = "high"
        elif utilization > 5:
            status = "active"
        else:
            status = "idle"
        gpus.append(
            {
                "id": int(idx),
                "name": name,
                "utilization": utilization,
                "memoryUsed": int(mem_used),
                "memoryTotal": int(mem_total),
                "temperature": int(temp),
                "status": status,
            }
        )
    return gpus


# ── Eval run status ─────────────────────────────────────────────


@app.get("/api/run/status")
def get_run_status() -> dict:
    """Return current eval.sh run status from .eval_status.json."""
    status_path = os.path.join(RESULT_DIR, ".eval_status.json")
    if not os.path.isfile(status_path):
        return {"running": False}
    try:
        with open(status_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"running": False}


@app.get("/api/run/results")
def get_run_results() -> dict:
    """Build a task×model results matrix from the result directory.

    Each entry reports ``pass`` (evaluation or prediction exists),
    ``fail`` (logged in eval_failures.log or error_message.jsonl),
    or ``running`` (currently executing according to .eval_status.json).
    """
    results: list[dict] = []
    seen: set[tuple[str, str]] = set()

    # 1. Walk result directory for completed runs (handles org/model nesting)
    if os.path.isdir(RESULT_DIR):
        for task_id in sorted(os.listdir(RESULT_DIR)):
            task_path = os.path.join(RESULT_DIR, task_id)
            if not os.path.isdir(task_path):
                continue
            for model_id, model_path in _iter_model_dirs(task_path):
                has_eval = os.path.isfile(os.path.join(model_path, "evaluation.jsonl"))
                has_pred = os.path.isfile(os.path.join(model_path, "prediction.jsonl"))
                has_error = os.path.isfile(os.path.join(model_path, "error_message.jsonl"))
                if has_eval or has_pred:
                    results.append({"task": task_id, "model": model_id, "status": "pass"})
                    seen.add((task_id, model_id))
                elif has_error:
                    results.append({"task": task_id, "model": model_id, "status": "fail"})
                    seen.add((task_id, model_id))

    # 2. Read failures log (FAIL|task|model|backend)
    fail_log = os.path.join(RESULT_DIR, "eval_failures.log")
    if os.path.isfile(fail_log):
        try:
            with open(fail_log) as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("FAIL|"):
                        continue
                    parts = line.split("|")
                    if len(parts) >= 3:
                        task, model = parts[1], parts[2]
                        if (task, model) not in seen:
                            results.append({"task": task, "model": model, "status": "fail"})
                            seen.add((task, model))
        except OSError:
            pass

    # 3. Mark currently running entry
    status = get_run_status()
    if status.get("running") and status.get("currentTask") and status.get("currentModel"):
        ct, cm = status["currentTask"], status["currentModel"]
        found = False
        for r in results:
            if r["task"] == ct and r["model"] == cm:
                r["status"] = "running"
                found = True
                break
        if not found:
            results.append({"task": ct, "model": cm, "status": "running"})

    return {"results": results}


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
        for model_id, model_path in _iter_model_dirs(task_path):
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
                    results.append(
                        {
                            "task_id": task_id,
                            "model_id": model_id,
                            "metrics": [],
                            "created_at": "",
                        }
                    )
            else:
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
    for model_id, model_path in _iter_model_dirs(task_path):
        eval_path = os.path.join(model_path, "evaluation.jsonl")
        if not os.path.isfile(eval_path):
            continue
        try:
            metrics = load_evaluation(model_path)
            scores.append({"model_id": model_id, "metrics": metrics})
        except Exception:
            continue

    return {"task_id": task_id, "models": scores}
