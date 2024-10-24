# This script reads the evaluation results from the result_dir and records the metrics in a spread sheet like below:
# | model_id | japanese-heron-bench | ja-vlm-bench-in-the-wild | ja-vg-vqa-500 |
# |----------| conv | ...           | llm-as-a-judge |  rougeL | ... | rougeL  |
# | gpt4o    | 0.5  | ...           | 0.6            | 0.7     | ... | 32.0    |
# Usage: python record_metrics.py --result_dir results/run_0 --task_id_list japanese-heron-bench ja-vlm-bench-in-the-wild ja-vg-vqa-500

import glob
import pandas as pd
import json
import pathlib
import os
from collections import defaultdict
import argparse


def create_metrics_per_model(result_dir: str, task_id_list: list[str]) -> list[dict]:
    # Return a list of dictionaries:
    # [{"model_id": "gpt4o", "japanese-heron-bench": {"conv": 0.5, ..., "overall_rel": 0.6}}, ...]

    model_metrics = defaultdict(dict)  # {model_id: {task_id: {metric: value}}}
    for task_id in task_id_list:
        task_id_result_dir = f"{result_dir}/{task_id}"
        files = glob.glob(os.path.join(task_id_result_dir, "evaluation/*.jsonl"))
        for file in files:
            with open(file, "r") as f:
                model_id = pathlib.Path(file).stem
                data = json.loads(f.readline())
                model_metrics[model_id][task_id] = data
    result_dict_list = []  # [{"model_id": "gpt4o", "japanese-heron": {"accuracy": 0.5, "f1": 0.6}}]
    for model_id, task_metrics in model_metrics.items():
        result_dict = {"model_id": model_id}
        for task_id, metrics in task_metrics.items():
            result_dict[task_id] = metrics
        result_dict_list.append(result_dict)

    return result_dict_list


def create_spread_sheet(metrics_per_model: list[dict]):
    # Create a spread sheet from the metrics_per_model [{mode_id, task_id {metric: value}}, ...]
    df = pd.json_normalize(metrics_per_model)
    # round to 2 decimal places
    df = df.round(2)

    df.columns = pd.MultiIndex.from_tuples([tuple(c.split(".")) for c in df.columns])
    print(df)
    df.to_excel("metrics_per_model.xlsx")


parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, default="results/run_0")
parser.add_argument(
    "--task_id_list",
    nargs="+",
    default=["japanese-heron-bench", "ja-vlm-bench-in-the-wild", "ja-vg-vqa-500", "jmmmu"],
)
args = parser.parse_args()

if __name__ == "__main__":
    result_dir = args.result_dir
    task_id_list = args.task_id_list

    metrics_per_model = create_metrics_per_model(result_dir, task_id_list)
    # store as a jsonl file
    with open("metrics_per_model.jsonl", "w") as f:
        for item in metrics_per_model:
            f.write(json.dumps(item) + "\n")

    # save
    create_spread_sheet(metrics_per_model)
