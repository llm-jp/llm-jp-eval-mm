"""Command-line interface for eval-mm."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import eval_mm
import eval_mm.metrics
from eval_mm.models.generation_config import GenerationConfig
from eval_mm.runner import run_evaluation
from eval_mm.tasks.task import TaskConfig


def _add_model_adapter_path(adapter_dir: str | None) -> None:
    """Add model adapter directory to sys.path so adapters can be imported."""
    if adapter_dir is None:
        # Default: look for examples/ relative to cwd, then repo root
        candidates = [
            Path.cwd() / "examples",
            Path(__file__).resolve().parents[2] / "examples",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                adapter_dir = str(candidate)
                break

    if adapter_dir and adapter_dir not in sys.path:
        sys.path.insert(0, adapter_dir)


def _load_model(model_id: str, backend: str, vllm_kwargs: dict):
    """Instantiate a model from the given backend."""
    if backend == "vllm":
        from eval_mm.models.base_vlm import BaseVLM

        # Import base_vllm from adapter path (examples/)
        base_vllm_mod = importlib.import_module("base_vllm")
        VLLM = base_vllm_mod.VLLM
        return VLLM(
            model_id=model_id,
            gpu_memory_utilization=vllm_kwargs.get("gpu_memory_utilization", 0.95),
            tensor_parallel_size=vllm_kwargs.get("tensor_parallel_size", 1),
        )
    else:
        model_table = importlib.import_module("model_table")
        model_cls = model_table.get_class_from_model_id(model_id)
        return model_cls(model_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-mm",
        description="eval-mm: Evaluate multi-modal language models",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run subcommand ---
    run_parser = subparsers.add_parser("run", help="Run evaluation pipeline")
    run_parser.add_argument("--model_id", required=True)
    run_parser.add_argument(
        "--task_id",
        default="japanese-heron-bench",
        help=f"Available: {eval_mm.TaskRegistry().get_task_list()}",
    )
    run_parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["heron-bench"],
        help=f"Available: {eval_mm.ScorerRegistry().get_metric_list()}",
    )
    run_parser.add_argument("--judge_model", default="gpt-4o-2024-11-20")
    run_parser.add_argument("--batch_size_for_evaluation", type=int, default=10)
    run_parser.add_argument("--result_dir", default="result")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--inference_only", action="store_true")
    run_parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Inference backend",
    )
    run_parser.add_argument("--adapter_dir", default=None, help="Path to model adapter modules")

    # Generation config
    run_parser.add_argument("--max_new_tokens", type=int, default=256)
    run_parser.add_argument("--num_beams", type=int, default=1)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--top_p", type=float, default=1.0)
    run_parser.add_argument("--do_sample", action="store_true", default=False)
    run_parser.add_argument("--use_cache", action="store_true", default=True)

    # Task config
    run_parser.add_argument("--max_dataset_len", type=int)
    run_parser.add_argument("--rotate_choices", action="store_true")
    run_parser.add_argument("--random_choice", action="store_true")

    # vLLM-specific
    run_parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    run_parser.add_argument("--tensor_parallel_size", type=int, default=1)
    run_parser.add_argument("--max_model_len", type=int, default=None)

    # --- evaluate subcommand (prediction already exists) ---
    eval_parser = subparsers.add_parser(
        "evaluate", help="Score existing predictions"
    )
    eval_parser.add_argument("--model_id", required=True)
    eval_parser.add_argument("--task_id", required=True)
    eval_parser.add_argument("--metrics", type=str, nargs="+", required=True)
    eval_parser.add_argument("--judge_model", default="gpt-4o-2024-11-20")
    eval_parser.add_argument("--batch_size_for_evaluation", type=int, default=10)
    eval_parser.add_argument("--result_dir", default="result")
    eval_parser.add_argument("--random_choice", action="store_true")

    # --- list subcommand ---
    list_parser = subparsers.add_parser("list", help="List available tasks and metrics")
    list_parser.add_argument(
        "what", choices=["tasks", "metrics"], help="What to list"
    )

    return parser


def cmd_run(args: argparse.Namespace) -> None:
    _add_model_adapter_path(args.adapter_dir)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        use_cache=args.use_cache,
    )
    task_config = TaskConfig(
        max_dataset_len=args.max_dataset_len,
        rotate_choices=args.rotate_choices,
    )

    vllm_kwargs = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
    }

    model = _load_model(args.model_id, args.backend, vllm_kwargs)

    run_evaluation(
        model=model,
        model_id=args.model_id,
        task_id=args.task_id,
        metrics=args.metrics,
        gen_config=gen_config,
        task_config=task_config,
        result_dir=args.result_dir,
        judge_model=args.judge_model,
        batch_size_for_evaluation=args.batch_size_for_evaluation,
        overwrite=args.overwrite,
        inference_only=args.inference_only,
        random_choice=args.random_choice,
        batch_mode=(args.backend == "vllm"),
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    task_config = TaskConfig()

    run_evaluation(
        model=None,
        model_id=args.model_id,
        task_id=args.task_id,
        metrics=args.metrics,
        task_config=task_config,
        result_dir=args.result_dir,
        judge_model=args.judge_model,
        batch_size_for_evaluation=args.batch_size_for_evaluation,
        random_choice=args.random_choice,
    )


def cmd_list(args: argparse.Namespace) -> None:
    if args.what == "tasks":
        tasks = eval_mm.TaskRegistry().get_task_list()
        print("Available tasks:")
        for t in sorted(tasks):
            print(f"  - {t}")
    elif args.what == "metrics":
        metrics = eval_mm.ScorerRegistry().get_metric_list()
        print("Available metrics:")
        for m in sorted(metrics):
            print(f"  - {m}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
