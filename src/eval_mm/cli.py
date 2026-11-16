"""Command-line interface for eval-mm."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import eval_mm
import eval_mm.metrics
from eval_mm.metadata import DEFAULT_JUDGE_MODEL
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


def _load_model(
    model_id: str, backend: str, vllm_kwargs: dict, openai_kwargs: dict | None = None
):
    """Instantiate a model from the given backend."""
    if backend == "openai":
        from eval_mm.models.openai_compat import OpenAICompatVLM

        return OpenAICompatVLM(model_id, **(openai_kwargs or {}))
    if backend == "vllm":

        # Import base_vllm from adapter path (examples/)
        base_vllm_mod = importlib.import_module("base_vllm")
        VLLM = base_vllm_mod.VLLM
        return VLLM(
            model_id=model_id,
            gpu_memory_utilization=vllm_kwargs.get("gpu_memory_utilization", 0.95),
            tensor_parallel_size=vllm_kwargs.get("tensor_parallel_size", 1),
            max_model_len=vllm_kwargs.get("max_model_len"),
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
        help=f"Available: {eval_mm.TaskRegistry.get_task_list()}",
    )
    run_parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["heron-bench"],
        help=f"Available: {eval_mm.ScorerRegistry.get_metric_list()}",
    )
    run_parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    run_parser.add_argument("--batch_size_for_evaluation", type=int, default=10)
    run_parser.add_argument("--result_dir", default="result")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--inference_only", action="store_true")
    run_parser.add_argument(
        "--backend",
        choices=["transformers", "vllm", "openai"],
        default="transformers",
        help="Inference backend. 'openai' talks to an OpenAI-compatible server "
        "(vllm serve, SGLang, or the OpenAI API) and needs no model adapter.",
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

    # OpenAI-compatible server
    run_parser.add_argument(
        "--base_url",
        default=None,
        help="Server URL, e.g. http://localhost:8000/v1. Omit to use the OpenAI API.",
    )
    run_parser.add_argument(
        "--api_key_env",
        default="OPENAI_API_KEY",
        help="Environment variable holding the API key.",
    )
    run_parser.add_argument(
        "--served_model_name",
        default=None,
        help="Model name the server expects, if it differs from --model_id.",
    )
    run_parser.add_argument("--concurrency", type=int, default=16)
    run_parser.add_argument(
        "--thinking",
        choices=["on", "off"],
        default=None,
        help="Set chat_template_kwargs.enable_thinking. Omit to keep the model default.",
    )
    run_parser.add_argument(
        "--reasoning_effort",
        default=None,
        help="reasoning_effort for APIs that take it.",
    )
    run_parser.add_argument(
        "--no_sampling_params",
        action="store_true",
        help="Do not send temperature/top_p (for APIs that reject them).",
    )

    # --- evaluate subcommand (prediction already exists) ---
    eval_parser = subparsers.add_parser(
        "evaluate", help="Score existing predictions"
    )
    eval_parser.add_argument("--model_id", required=True)
    eval_parser.add_argument("--task_id", required=True)
    eval_parser.add_argument("--metrics", type=str, nargs="+", required=True)
    eval_parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    eval_parser.add_argument("--batch_size_for_evaluation", type=int, default=10)
    eval_parser.add_argument("--result_dir", default="result")
    eval_parser.add_argument("--random_choice", action="store_true")

    # --- run-model subcommand (serve a pool model and run tasks against it) ---
    pool_parser = subparsers.add_parser(
        "run-model",
        help="Serve models from the model pool and evaluate them on several tasks",
    )
    target = pool_parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model_id", nargs="+", help="Model IDs in the model pool")
    target.add_argument("--all", action="store_true", help="Every active model in the pool")
    pool_parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task IDs. Default: every task with a default metric.",
    )
    pool_parser.add_argument(
        "--base_url",
        default=None,
        help="Use a server that is already running instead of starting one.",
    )
    pool_parser.add_argument(
        "--vllm_cmd",
        default=os.getenv("EVAL_MM_VLLM_CMD", "vllm"),
        help="Command that runs vLLM, e.g. 'uvx --from vllm==0.33.0 vllm'. "
        "Defaults to $EVAL_MM_VLLM_CMD or 'vllm'.",
    )
    pool_parser.add_argument("--port", type=int, default=8000)
    pool_parser.add_argument("--startup_timeout", type=float, default=3600)
    pool_parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    pool_parser.add_argument("--concurrency", type=int, default=16)
    pool_parser.add_argument(
        "--thinking",
        choices=["on", "off"],
        default=None,
        help="Override the pool's thinking setting.",
    )
    pool_parser.add_argument("--max_new_tokens", type=int, default=1024)
    pool_parser.add_argument("--max_dataset_len", type=int)
    pool_parser.add_argument("--result_dir", default="result")
    pool_parser.add_argument("--log_dir", default="logs")
    pool_parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    pool_parser.add_argument("--overwrite", action="store_true")
    pool_parser.add_argument(
        "--inference_only",
        action="store_true",
        help="Generate predictions only; score later with `eval-mm evaluate`.",
    )

    # --- list subcommand ---
    list_parser = subparsers.add_parser(
        "list", help="List available tasks, metrics and pool models"
    )
    list_parser.add_argument(
        "what", choices=["tasks", "metrics", "models"], help="What to list"
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

    openai_kwargs = {
        "base_url": args.base_url,
        "api_key": os.getenv(args.api_key_env),
        "served_model_name": args.served_model_name,
        "concurrency": args.concurrency,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
        "send_sampling_params": not args.no_sampling_params,
    }

    model = _load_model(args.model_id, args.backend, vllm_kwargs, openai_kwargs)

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
        batch_mode=(args.backend in ("vllm", "openai")),
        batch_chunk_size=args.concurrency * 4 if args.backend == "openai" else 0,
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


def cmd_run_model(args: argparse.Namespace) -> int:
    from contextlib import nullcontext
    from pathlib import Path

    from eval_mm.metadata import TASKS
    from eval_mm.model_pool import get_spec, load_pool
    from eval_mm.models.openai_compat import OpenAICompatVLM
    from eval_mm.serve import serving

    if args.all:
        specs = [s for s in load_pool() if s.status == "active"]
    else:
        specs = [get_spec(model_id) for model_id in args.model_id]

    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_ids = [t for t, meta in TASKS.items() if meta.default_metrics]
    registered = set(eval_mm.TaskRegistry.get_task_list())
    unknown = [t for t in task_ids if t not in registered]
    if unknown:
        raise SystemExit(f"Unknown tasks: {unknown}")

    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens)
    failures: list[str] = []

    for spec in specs:
        if spec.source == "api" or args.base_url:
            server = nullcontext(args.base_url)
        else:
            log_path = Path(args.log_dir) / f"serve-{spec.id.replace('/', '__')}.log"
            server = serving(
                spec,
                port=args.port,
                vllm_cmd=args.vllm_cmd,
                log_path=log_path,
                startup_timeout=args.startup_timeout,
            )
        try:
            with server as base_url:
                model = OpenAICompatVLM(
                    spec.id,
                    base_url=base_url,
                    api_key=os.getenv(args.api_key_env),
                    concurrency=args.concurrency,
                    thinking=args.thinking or spec.thinking,
                )
                for task_id in task_ids:
                    metrics = TASKS[task_id].default_metrics if task_id in TASKS else []
                    try:
                        run_evaluation(
                            model=model,
                            model_id=spec.id,
                            task_id=task_id,
                            metrics=metrics,
                            gen_config=gen_config,
                            task_config=TaskConfig(max_dataset_len=args.max_dataset_len),
                            result_dir=args.result_dir,
                            judge_model=args.judge_model,
                            overwrite=args.overwrite,
                            inference_only=args.inference_only or not metrics,
                            batch_mode=True,
                            batch_chunk_size=args.concurrency * 4,
                        )
                    except Exception as e:  # noqa: BLE001 - keep going with the other tasks
                        print(f"FAIL {spec.id} {task_id}: {type(e).__name__}: {e}", file=sys.stderr)
                        failures.append(f"{spec.id}|{task_id}")
        except Exception as e:  # noqa: BLE001 - keep going with the other models
            print(f"FAIL {spec.id} (server): {type(e).__name__}: {e}", file=sys.stderr)
            failures.append(f"{spec.id}|server")

    if failures:
        print(f"{len(failures)} failed: {failures}", file=sys.stderr)
        return 1
    return 0


def cmd_list(args: argparse.Namespace) -> None:
    if args.what == "tasks":
        tasks = eval_mm.TaskRegistry.get_task_list()
        print("Available tasks:")
        for t in sorted(tasks):
            print(f"  - {t}")
    elif args.what == "metrics":
        metrics = eval_mm.ScorerRegistry.get_metric_list()
        print("Available metrics:")
        for m in sorted(metrics):
            print(f"  - {m}")
    elif args.what == "models":
        from eval_mm.model_pool import load_pool

        for spec in load_pool():
            thinking = f" thinking={spec.thinking}" if spec.thinking else ""
            print(f"{spec.status:8s} {spec.id} (gpus={spec.gpus}{thinking})")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "run-model":
        sys.exit(cmd_run_model(args))
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
