"""Batch-run judge-only scoring across all (task, model) cells using a
locally-loaded vLLM judge (e.g. ``openai/gpt-oss-20b``).

Skips cells that:
  * don't have ``prediction.jsonl``
  * already have ``evaluation.jsonl`` (unless ``--overwrite``)
  * match a skip list of tasks (multi-image / missing-pred tasks)

The vLLM judge is loaded **once** and reused across every cell.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# task_id -> metric (mirrors tsubame/run_model.sh)
METRIC_MAP: dict[str, str] = {
    "japanese-heron-bench": "heron-bench",
    "ja-vlm-bench-in-the-wild": "llm-as-a-judge",
    "ja-vg-vqa-500": "llm-as-a-judge",
    "jmmmu": "jmmmu",
    "ja-multi-image-vqa": "llm-as-a-judge",
    "jdocqa": "llm-as-a-judge",
    "mmmu": "mmmu",
    "llava-bench-in-the-wild": "llm-as-a-judge",
    "mecha-ja": "mecha-ja",
    "cc-ocr": "cc-ocr",
    "ai2d": "ai2d",
    "cvqa": "substring-match",
    "docvqa": "substring-match",
    "infographicvqa": "substring-match",
    "textvqa": "substring-match",
    "chartqa": "substring-match",
    "chartqapro": "substring-match",
    "okvqa": "substring-match",
    "jawildtext-board-vqa": "jawildtext-board-vqa",
    "jawildtext-handwriting-ocr": "jawildtext-handwriting-ocr",
    "jawildtext-receipt-kie": "jawildtext-receipt-kie",
    "mathvista": "mathvista",
}

# Metrics that actually hit the judge client.
LLM_METRICS = {"llm-as-a-judge", "heron-bench", "jawildtext-board-vqa"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", default="/gs/bs/tga-okazaki/maeda/eval-mm-results")
    p.add_argument("--judge_model", default="openai/gpt-oss-20b")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--reasoning_effort", default="low",
                   help="gpt-oss reasoning effort: low/medium/high (or 'none' to disable)")
    p.add_argument("--batch_size", type=int, default=512,
                   help="per-scorer batch (vLLM still batches internally)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--model_filter", default="",
                   help="substring match on 'org/model' id")
    p.add_argument("--task_filter", default="",
                   help="comma-separated task ids to include")
    p.add_argument("--skip_tasks",
                   default="ja-multi-image-vqa,jdocqa,jmmmu,mmmu,mathvista",
                   help="comma-separated task ids to exclude")
    p.add_argument("--local_only", action="store_true",
                   help="only run metrics that don't need the LLM judge (fast)")
    p.add_argument("--judge_only", action="store_true",
                   help="only run metrics that need the LLM judge")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after N cells (for smoke test)")
    return p.parse_args()


def build_plan(args) -> list[tuple[str, str, str]]:
    skip = {t.strip() for t in args.skip_tasks.split(",") if t.strip()}
    task_filter = {t.strip() for t in args.task_filter.split(",") if t.strip()}
    rd = Path(args.result_dir)
    plan: list[tuple[str, str, str]] = []
    for task_dir in sorted(rd.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        if task.startswith("eval_failures_"):
            continue
        if task in skip:
            continue
        if task_filter and task not in task_filter:
            continue
        metric = METRIC_MAP.get(task, "substring-match")
        is_llm = metric in LLM_METRICS
        if args.local_only and is_llm:
            continue
        if args.judge_only and not is_llm:
            continue
        for org_dir in sorted(task_dir.iterdir()):
            if not org_dir.is_dir():
                continue
            for model_dir in sorted(org_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model_id = f"{org_dir.name}/{model_dir.name}"
                if args.model_filter and args.model_filter not in model_id:
                    continue
                if not (model_dir / "prediction.jsonl").exists():
                    continue
                if (model_dir / "evaluation.jsonl").exists() and not args.overwrite:
                    continue
                plan.append((task, model_id, metric))
    return plan


def main() -> int:
    args = parse_args()

    plan = build_plan(args)
    llm_cells = sum(1 for _, _, m in plan if m in LLM_METRICS)
    local_cells = len(plan) - llm_cells

    print(f"Planned cells: {len(plan)} (LLM judge={llm_cells}, local={local_cells})")
    if args.limit and len(plan) > args.limit:
        print(f"Limiting to first {args.limit}")
        plan = plan[: args.limit]

    for i, (task, model, metric) in enumerate(plan, 1):
        print(f"  [{i}] {task} :: {model}  ({metric})")

    if args.dry_run or not plan:
        return 0

    # Init vLLM only if at least one LLM-judge cell is planned
    client = None
    need_llm = any(m in LLM_METRICS for _, _, m in plan)
    if need_llm:
        from eval_mm.utils.vllm_judge_client import VLLMChatAPI
        client = VLLMChatAPI(
            model_id=args.judge_model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            reasoning_effort=(
                None if args.reasoning_effort.lower() == "none"
                else args.reasoning_effort
            ),
        )
    else:
        print("No LLM-judge cells — running local-only scoring.")

    from eval_mm import TaskConfig, run_evaluation

    fail_log = Path(args.result_dir) / "judge_failures_vllm.log"
    n_ok = 0
    n_fail = 0
    for i, (task, model_id, metric) in enumerate(plan, 1):
        print(f"\n===== [{i}/{len(plan)}] {task} :: {model_id} ({metric}) =====")
        try:
            run_evaluation(
                model=None,
                model_id=model_id,
                task_id=task,
                metrics=[metric],
                task_config=TaskConfig(),
                result_dir=args.result_dir,
                judge_model=args.judge_model,
                batch_size_for_evaluation=args.batch_size,
                client=client,
            )
            n_ok += 1
        except Exception as e:
            n_fail += 1
            msg = f"FAIL|{task}|{model_id}|{type(e).__name__}: {e}"
            print(msg, file=sys.stderr)
            traceback.print_exc()
            with open(fail_log, "a") as fh:
                fh.write(msg + "\n")

    print(f"\n========== Done: ok={n_ok}, fail={n_fail}, total={len(plan)} ==========")
    if n_fail:
        print(f"Failures logged to: {fail_log}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
