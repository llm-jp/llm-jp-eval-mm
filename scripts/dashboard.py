"""Benchmark Runner Dashboard.

Launch:
    uv run streamlit run scripts/dashboard.py -- --result_dir result

Provides:
- GPU monitoring (utilization, memory, temperature)
- Available models / tasks matrix with completion status
- Evaluation launch controls
- Live progress tracking
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import streamlit as st

# -- Setup paths so eval_mm and examples are importable --
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "examples"))

import eval_mm
import eval_mm.metrics
from eval_mm import load_evaluation
from eval_mm.metadata import TASKS, METRICS, LEADERBOARD_MODELS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULT_DIR = "result"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="result")
    return parser.parse_args()


def get_gpu_info() -> list[dict]:
    """Query nvidia-smi for GPU status."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    gpus = []
    for line in out.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util_pct": int(parts[2]),
                    "mem_used_mb": int(parts[3]),
                    "mem_total_mb": int(parts[4]),
                    "temp_c": int(parts[5]),
                }
            )
    return gpus


def get_available_models() -> list[str]:
    """Models registered in examples/model_table.py."""
    try:
        from model_table import MODEL_ID_TO_CLASS_PATH

        return sorted(MODEL_ID_TO_CLASS_PATH.keys())
    except ImportError:
        return sorted(LEADERBOARD_MODELS)


def get_available_tasks() -> list[str]:
    return sorted(eval_mm.TaskRegistry.get_task_list())


def get_available_metrics() -> list[str]:
    return sorted(eval_mm.ScorerRegistry.get_metric_list())


def get_default_metrics(task_id: str) -> list[str]:
    """Return default metrics for a task from metadata."""
    meta = TASKS.get(task_id)
    return meta.default_metrics if meta else []


def scan_results(result_dir: str) -> dict[tuple[str, str], list[str]]:
    """Scan result_dir and return {(task, model): [available_files]}."""
    results: dict[tuple[str, str], list[str]] = {}
    if not os.path.isdir(result_dir):
        return results
    for task_id in sorted(os.listdir(result_dir)):
        task_path = os.path.join(result_dir, task_id)
        if not os.path.isdir(task_path):
            continue
        for model_dir_root in sorted(os.listdir(task_path)):
            model_root_path = os.path.join(task_path, model_dir_root)
            if not os.path.isdir(model_root_path):
                continue
            # Handle nested model dirs like Qwen/Qwen2.5-VL-3B-Instruct
            for sub in sorted(os.listdir(model_root_path)):
                sub_path = os.path.join(model_root_path, sub)
                if os.path.isdir(sub_path):
                    model_id = f"{model_dir_root}/{sub}"
                    files = os.listdir(sub_path)
                    results[(task_id, model_id)] = files
                elif sub == "prediction.jsonl":
                    # Direct model dir (no nesting)
                    model_id = model_dir_root
                    files = os.listdir(model_root_path)
                    results[(task_id, model_id)] = files
                    break
    return results


def load_evaluation_score(result_dir: str, task_id: str, model_id: str) -> dict | None:
    """Load evaluation.jsonl for a (task, model) pair via result_schema API."""
    output_dir = os.path.join(result_dir, task_id, model_id)
    if not os.path.isfile(os.path.join(output_dir, "evaluation.jsonl")):
        return None
    try:
        return load_evaluation(output_dir)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session state for background processes
# ---------------------------------------------------------------------------

if "running_jobs" not in st.session_state:
    st.session_state.running_jobs = {}  # {job_id: {info}}


def start_evaluation(
    model_id: str,
    task_id: str,
    metrics: list[str],
    backend: str,
    gpu_id: int,
    result_dir: str,
    max_new_tokens: int,
    overwrite: bool,
):
    """Launch evaluation as a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "eval_mm",
        "run",
        "--backend",
        backend,
        "--model_id",
        model_id,
        "--task_id",
        task_id,
        "--metrics",
        *metrics,
        "--result_dir",
        result_dir,
        "--max_new_tokens",
        str(max_new_tokens),
    ]
    if overwrite:
        cmd.append("--overwrite")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = f"/tmp/eval_mm_job_{model_id.replace('/', '_')}_{task_id}.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(_repo_root),
    )

    job_id = f"{model_id}:{task_id}:{time.time():.0f}"
    st.session_state.running_jobs[job_id] = {
        "model_id": model_id,
        "task_id": task_id,
        "metrics": metrics,
        "gpu_id": gpu_id,
        "pid": proc.pid,
        "proc": proc,
        "log_path": log_path,
        "started_at": time.strftime("%H:%M:%S"),
    }
    return job_id


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    result_dir = args.result_dir

    st.set_page_config(page_title="eval-mm Dashboard", layout="wide")
    st.title("eval-mm Dashboard")

    # ── GPU Status ──────────────────────────────────────────────
    st.header("GPU Status")
    gpus = get_gpu_info()
    if gpus:
        cols = st.columns(len(gpus))
        for i, gpu in enumerate(gpus):
            with cols[i]:
                mem_pct = gpu["mem_used_mb"] / gpu["mem_total_mb"] * 100
                st.metric(
                    f"GPU {gpu['index']}: {gpu['name']}",
                    f"{gpu['util_pct']}% util",
                )
                st.progress(gpu["util_pct"] / 100)
                st.caption(
                    f"Memory: {gpu['mem_used_mb']}/{gpu['mem_total_mb']} MB ({mem_pct:.0f}%) | {gpu['temp_c']}°C"
                )
    else:
        st.warning("GPU not detected")

    st.divider()

    # ── Running Jobs ────────────────────────────────────────────
    st.header("Running Jobs")
    if st.session_state.running_jobs:
        for job_id, job in list(st.session_state.running_jobs.items()):
            proc = job["proc"]
            poll = proc.poll()
            if poll is None:
                status = "running"
                icon = "🔄"
            elif poll == 0:
                status = "completed"
                icon = "✅"
            else:
                status = f"failed (exit {poll})"
                icon = "❌"

            with st.expander(
                f"{icon} {job['model_id']} × {job['task_id']} — GPU {job['gpu_id']} [{job['started_at']}] — {status}",
                expanded=(poll is None),
            ):
                if os.path.isfile(job["log_path"]):
                    with open(job["log_path"]) as f:
                        lines = f.readlines()
                    # Show last 20 lines
                    st.code("".join(lines[-20:]), language="text")
                if poll is not None:
                    if st.button(f"Remove", key=f"rm_{job_id}"):
                        del st.session_state.running_jobs[job_id]
                        st.rerun()
    else:
        st.caption("No running jobs")

    st.divider()

    # ── Launch Evaluation ───────────────────────────────────────
    st.header("Launch Evaluation")

    col_left, col_right = st.columns(2)

    with col_left:
        available_models = get_available_models()
        selected_model = st.selectbox("Model", available_models, index=0)

        selected_task = st.selectbox("Task", get_available_tasks(), index=0)

        defaults = get_default_metrics(selected_task)
        all_metrics = get_available_metrics()
        selected_metrics = st.multiselect(
            "Metrics",
            all_metrics,
            default=[m for m in defaults if m in all_metrics] or [all_metrics[0]],
        )

    with col_right:
        backend = st.radio("Backend", ["transformers", "vllm"], horizontal=True)

        gpu_options = [f"GPU {g['index']}" for g in gpus] if gpus else ["GPU 0"]
        selected_gpu = st.selectbox("GPU", gpu_options)
        gpu_id = int(selected_gpu.split()[-1])

        max_new_tokens = st.slider("Max new tokens", 32, 1024, 256)
        overwrite = st.checkbox("Overwrite existing predictions")

    # Check if result already exists
    eval_result = load_evaluation_score(result_dir, selected_task, selected_model)
    if eval_result and not overwrite:
        st.info(f"Result already exists for {selected_model} × {selected_task}")
        st.json(eval_result)

    if st.button("Start Evaluation", type="primary", disabled=(not selected_metrics)):
        job_id = start_evaluation(
            selected_model,
            selected_task,
            selected_metrics,
            backend,
            gpu_id,
            result_dir,
            max_new_tokens,
            overwrite,
        )
        st.success(f"Started: {job_id}")
        st.rerun()

    st.divider()

    # ── Results Matrix ──────────────────────────────────────────
    st.header("Results Matrix")

    existing = scan_results(result_dir)
    tasks_with_results = sorted({t for t, m in existing.keys()})
    models_with_results = sorted({m for t, m in existing.keys()})

    if tasks_with_results and models_with_results:
        # Build matrix
        import pandas as pd

        matrix_data = {}
        for model_id in models_with_results:
            row = {}
            for task_id in tasks_with_results:
                files = existing.get((task_id, model_id), [])
                has_pred = "prediction.jsonl" in files
                has_eval = "evaluation.jsonl" in files
                if has_eval:
                    ev = load_evaluation_score(result_dir, task_id, model_id)
                    if ev:
                        # Show first metric's overall score
                        first_metric = next(iter(ev), None)
                        if first_metric and "overall_score" in ev[first_metric]:
                            score = ev[first_metric]["overall_score"]
                            row[task_id] = f"{score:.2f}" if isinstance(score, float) else str(score)
                        else:
                            row[task_id] = "✅"
                    else:
                        row[task_id] = "✅"
                elif has_pred:
                    row[task_id] = "📝"
                else:
                    row[task_id] = ""
            matrix_data[model_id] = row

        df = pd.DataFrame.from_dict(matrix_data, orient="index")
        # Apply display names from metadata
        rename = {t.task_id: t.display_name for t in TASKS.values() if t.task_id in df.columns}
        df = df.rename(columns=rename)

        st.caption("✅ = evaluated | 📝 = predictions only | score = overall score")
        st.dataframe(df, use_container_width=True)
    else:
        st.caption(f"No results found in `{result_dir}/`")

    # ── Auto-refresh ────────────────────────────────────────────
    has_running = any(
        job["proc"].poll() is None for job in st.session_state.running_jobs.values()
    )
    if has_running:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
