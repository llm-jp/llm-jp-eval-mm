"""Evaluate a VLM using the transformers backend.

This is a thin wrapper around ``eval_mm.run_evaluation``.
For the canonical CLI, use: ``python -m eval_mm run --backend transformers ...``
"""

import argparse

import eval_mm
import eval_mm.metrics
from eval_mm import GenerationConfig, TaskConfig, run_evaluation
from model_table import get_class_from_model_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument(
        "--task_id",
        default="japanese-heron-bench",
        help=f"Task ID to evaluate. Available: {eval_mm.TaskRegistry().get_task_list()}",
    )
    parser.add_argument("--judge_model", default="gpt-4o-2024-11-20")
    parser.add_argument("--batch_size_for_evaluation", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--max_dataset_len", type=int)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["heron-bench"],
        help=f"Metrics to evaluate. Available: {eval_mm.ScorerRegistry().get_metric_list()}",
    )
    parser.add_argument("--rotate_choices", action="store_true")
    parser.add_argument("--random_choice", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

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

    model = get_class_from_model_id(args.model_id)(args.model_id)

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
        batch_mode=False,
    )


if __name__ == "__main__":
    main()
