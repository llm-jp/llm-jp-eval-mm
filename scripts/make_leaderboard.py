import json
import os
import pandas as pd
from argparse import ArgumentParser
from typing import List, Optional
from loguru import logger
import eval_mm
import eval_mm.metrics

# {"llm_as_a_judge_heron_bench": {"overall_score": 36.9037848990212, "details": {"conv": 3.238095238095238, "conv_rel": 35.78947368421053, "detail": 3.1904761904761907, "detail_rel": 36.61202185792351, "complex": 3.4, "complex_rel": 38.309859154929576, "parse_error_count": 0, "overall": 3.29126213592233, "overall_rel": 36.9037848990212}}}


TASK_ALIAS = {
    "japanese-heron-bench": "Heron",
    "ja-vlm-bench-in-the-wild": "JVB-ItW",
    "ja-vg-vqa-500": "VG-VQA",
    "jdocqa": "JDocQA",
    "ja-multi-image-vqa": "MulIm-VQA",
    "jmmmu": "JMMMU",
    "jic-vqa": "JIC",
    "mecha-ja": "Mecha",
    "llava-bench-in-the-wild": "LLAVA",
    "mmmu": "MMMU",
}

METRIC_ALIAS = {
    "heron-bench": "LLM",
    "llm-as-a-judge": "LLM",
    "rougel": "Rouge",
    "jdocqa": "Acc",
    "jmmmu": "Acc",
    "jic-vqa": "Acc",
    "mecha-ja": "Acc",
    "mmmu": "Acc",
}


def main(
    result_dir: str,
    model_list: List[str],
    output_path: Optional[str] = None,
    output_format: str = "markdown",
):
    task_dirs = [d for d in os.listdir(result_dir) if not d.startswith(".")]

    df = pd.DataFrame()

    for model in model_list:
        model_results = {"Model": model}

        for task_dir in task_dirs:
            logger.info(f"Reading {task_dir} evaluation results for {model}")
            eval_path = os.path.join(result_dir, task_dir, model, "evaluation.jsonl")
            if not os.path.exists(eval_path):
                logger.warning(f"{eval_path} not found. Skipping...")
                continue

            with open(eval_path, "r") as f:
                evaluation = json.load(f)
            logger.info(f"Results: {evaluation}")
            # {'llm_as_a_judge': {'overall_score': 2.7, 'details': {'llm_as_a_judge': 2.7}}, 'rougel': {'overall_score': 40.75248314834134, 'details': {'rougel': 40.75248314834134}}}

            for metric, aggregate_output in evaluation.items():
                if metric not in list(eval_mm.metrics.ScorerRegistry._scorers.keys()):
                    logger.warning(f"Skipping unsupported metric: {metric}")
                    continue

                model_results[f"{task_dir}/{metric}"] = aggregate_output[
                    "overall_score"
                ]

        df = df._append(model_results, ignore_index=True)

    df = df.set_index("Model")
    # round to 2 decimal places
    df = df.round(2)
    df = df.rename(
        columns={
            k: f"{TASK_ALIAS[k.split('/')[0]]}/{METRIC_ALIAS[k.split('/')[1]]}"
            for k in df.columns
        }
    )
    # sort columns
    df = df.reindex(sorted(df.columns), axis=1)
    if output_format == "markdown":
        table = df.to_markdown(mode="github", floatfmt=".2f")
    elif output_format == "latex":
        table = df.to_latex(float_format="%.2f")
    print(table)

    with open(output_path, "w") as f:
        f.write(table)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--output_path", type=str, default="leaderboard.md")
    parser.add_argument(
        "--output_format", type=str, default="markdown", choices=["markdown", "latex"]
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # モデルは実行時引数でも取れるようにしても良い
    model_list = [
        "stabilityai/japanese-instructblip-alpha",
        "stabilityai/japanese-stable-vlm",
        "SakanaAI/Llama-3-EvoVLM-JP-v2",
        "cyberagent/llava-calm2-siglip",
        "llm-jp/llm-jp-3-vila-14b",
        "sbintuitions/sarashina2-vision-8b",
        "sbintuitions/sarashina2-vision-14b",
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "neulab/Pangea-7B-hf",
        "mistralai/Pixtral-12B-2409",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "Efficient-Large-Model/VILA1.5-13b",
        "OpenGVLab/InternVL2-8B",
        "OpenGVLab/InternVL2-26B",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "microsoft/Phi-4-multimodal-instruct",
        "gpt-4o-2024-11-20",
    ]

    main(args.result_dir, model_list, args.output_path, args.output_format)
