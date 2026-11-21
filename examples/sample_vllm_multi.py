"""Run all tasks for a single model using the vLLM backend.

The model is loaded **once** and reused across tasks, keeping GPU utilization
high and avoiding repeated load/unload overhead.

Designed to be called from eval.sh with::

    python examples/sample_vllm_multi.py \
        --model_id Qwen/Qwen2-VL-7B-Instruct \
        --task_ids "jmmmu,japanese-heron-bench,mmmu" \
        --metrics "jmmmu,heron-bench,mmmu" \
        --status_file result/.eval_status.json \
        --completed_offset 42 --failed_offset 0 --total_runs 1064 \
        --start_epoch 1718000000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import eval_mm
import eval_mm.metrics
from eval_mm import GenerationConfig, TaskConfig, run_evaluation
from base_vllm import VLLM


def parse_args():
    p = argparse.ArgumentParser(description="Multi-task vLLM evaluation")
    p.add_argument("--model_id", required=True)
    p.add_argument("--task_ids", required=True, help="Comma-separated task IDs")
    p.add_argument("--metrics", required=True, help="Comma-separated metrics (same order as tasks)")
    p.add_argument("--result_dir", default="result")
    p.add_argument("--judge_model", default="gpt-5.1-2025-11-13")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--inference_only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_chunk_size", type=int, default=50,
                   help="Process inference in chunks of this size for progress tracking")
    # Status tracking (passed from eval.sh)
    p.add_argument("--status_file", default="")
    p.add_argument("--fail_log", default="")
    p.add_argument("--completed_offset", type=int, default=0)
    p.add_argument("--failed_offset", type=int, default=0)
    p.add_argument("--total_runs", type=int, default=0)
    p.add_argument("--start_epoch", type=int, default=0)
    return p.parse_args()


def write_status(args, task_id: str, completed: int, failed: int,
                 inference: dict | None = None):
    """Write eval status JSON compatible with the web dashboard."""
    if not args.status_file:
        return
    now = int(time.time())
    elapsed = now - args.start_epoch if args.start_epoch else 0
    eta = 0
    if completed > 0:
        eta = int(elapsed * (args.total_runs - completed) / completed)
    status = {
        "running": True,
        "currentTask": task_id,
        "currentModel": args.model_id,
        "backend": "vllm",
        "completed": completed,
        "failed": failed,
        "total": args.total_runs,
        "progress": int(completed * 100 / args.total_runs) if args.total_runs else 0,
        "etaSeconds": eta,
        "elapsedSeconds": elapsed,
    }
    if inference:
        status["inference"] = inference
    try:
        with open(args.status_file, "w") as f:
            json.dump(status, f, indent=2)
    except OSError:
        pass


def main():
    args = parse_args()

    tasks = args.task_ids.split(",")
    metrics_list = args.metrics.split(",")
    assert len(tasks) == len(metrics_list), (
        f"task count ({len(tasks)}) != metric count ({len(metrics_list)})"
    )

    completed = args.completed_offset
    failed = args.failed_offset

    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens)
    task_config = TaskConfig()

    # ── Load model ONCE ──────────────────────────────────────────
    write_status(args, tasks[0], completed, failed,
                 inference={"current": 0, "total": 0, "phase": "loading_model"})

    try:
        model = VLLM(
            model_id=args.model_id,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
    except Exception as e:
        print(f"FAIL|model_load|{args.model_id}|vllm|{e}")
        # Mark all tasks for this model as failed
        for task_id in tasks:
            if args.fail_log:
                try:
                    with open(args.fail_log, "a") as f:
                        f.write(f"FAIL|{task_id}|{args.model_id}|vllm\n")
                except OSError:
                    pass
            failed += 1
            completed += 1
        write_status(args, tasks[-1], completed, failed)
        return

    # ── Run all tasks ────────────────────────────────────────────
    for task_id, metric in zip(tasks, metrics_list):
        write_status(args, task_id, completed, failed,
                     inference={"current": 0, "total": 0, "phase": "loading_dataset"})

        def progress_cb(current: int, total: int):
            write_status(args, task_id, completed, failed,
                         inference={"current": current, "total": total, "phase": "inferring"})

        try:
            run_evaluation(
                model=model,
                model_id=args.model_id,
                task_id=task_id,
                metrics=[metric],
                gen_config=gen_config,
                task_config=task_config,
                result_dir=args.result_dir,
                judge_model=args.judge_model,
                inference_only=args.inference_only,
                overwrite=args.overwrite,
                batch_mode=True,
                batch_chunk_size=args.batch_chunk_size,
                progress_callback=progress_cb,
            )
        except Exception as e:
            print(f"FAIL|{task_id}|{args.model_id}|vllm|{e}")
            if args.fail_log:
                try:
                    with open(args.fail_log, "a") as f:
                        f.write(f"FAIL|{task_id}|{args.model_id}|vllm\n")
                except OSError:
                    pass
            failed += 1

        completed += 1
        write_status(args, task_id, completed, failed)

    # Write final status so eval.sh can read back counts
    write_status(args, tasks[-1], completed, failed)


if __name__ == "__main__":
    main()
