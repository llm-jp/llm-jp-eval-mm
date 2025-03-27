import json
import os
import pandas as pd
from argparse import ArgumentParser
from typing import List, Optional
from loguru import logger
import eval_mm
import eval_mm.metrics
import seaborn as sns
import matplotlib.pyplot as plt

TASK_ALIAS = {
    "japanese-heron-bench": "Heron",
    "ja-vlm-bench-in-the-wild": "JVB-ItW",
    "ja-vg-vqa-500": "VG-VQA",
    "jdocqa": "JDocQA",
    "ja-multi-image-vqa": "MulIm-VQA",
    "jmmmu": "JMMMU",
    "jic-vqa": "JIC",
    "mecha-ja": "MECHA",
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


def load_evaluation_data(result_dir: str, model: str, task_dirs: List[str]) -> dict:
    """Load evaluation results for a given model across multiple tasks."""
    model_results = {"Model": model}
    for task_dir in task_dirs:
        eval_path = os.path.join(result_dir, task_dir, model, "evaluation.jsonl")
        if not os.path.exists(eval_path):
            logger.warning(f"{eval_path} not found. Skipping...")
            continue

        with open(eval_path, "r") as f:
            evaluation = json.load(f)

        for metric, aggregate_output in evaluation.items():
            if metric not in eval_mm.metrics.ScorerRegistry._scorers.keys():
                logger.warning(f"Skipping unsupported metric: {metric}")
                continue

            model_results[f"{task_dir}/{metric}"] = aggregate_output["overall_score"]

    return model_results


def process_results(result_dir: str, model_list: List[str]) -> pd.DataFrame:
    """Process all evaluation results into a structured DataFrame."""
    task_dirs = [d for d in os.listdir(result_dir) if not d.startswith(".")]
    df = pd.DataFrame()

    for model in model_list:
        logger.info(f"Processing results for {model}")
        model_results = load_evaluation_data(result_dir, model, task_dirs)
        if not model_results:
            continue
        df = df._append(model_results, ignore_index=True)

    df = df.set_index("Model").round(2)
    df = df.rename(
        columns={
            k: f"{TASK_ALIAS[k.split('/')[0]]}/{METRIC_ALIAS[k.split('/')[1]]}"
            for k in df.columns
        }
    )
    return df


def generate_json_path(df: pd.DataFrame, output_path: str):
    """Generate JSON output from DataFrame and save it to a file."""
    json_data = []

    for model, row in df.iterrows():
        if pd.isna(row).all():
            continue
        model_entry = {
            "model": model,
            "url": f"https://huggingface.co/{model}",
            "scores": {},
        }

        for col, score in row.items():
            if isinstance(score, (int, float)) and not pd.isna(score):
                task, metric = col.split("/")
                if task not in model_entry["scores"]:
                    model_entry["scores"][task] = {}
                model_entry["scores"][task][metric] = score

        json_data.append(model_entry)

    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"JSON output saved to {output_path}")


def plot_correlation(df: pd.DataFrame, filename: str):
    """Plot and save the correlation matrix."""
    corr = df.corr(method="spearman")
    plt.figure(figsize=(12, 12))
    sns.clustermap(
        corr,
        method="average",
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        figsize=(12, 10),
        cbar_pos=(0.02, 0.8, 0.03, 0.18),
    )
    plt.tight_layout()
    plt.savefig(filename)


def plot_task_performance(df: pd.DataFrame):
    """Generate bar plots for each task."""
    df_plot = df.reset_index().melt(
        id_vars=["Model"], var_name="Task", value_name="Score"
    )

    for task in df_plot["Task"].unique():
        plt.figure(figsize=(10, 6))
        task_data = df_plot[df_plot["Task"] == task].sort_values(
            "Score", ascending=False
        )
        sns.barplot(data=task_data, x="Model", y="Score", palette="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.title(f"Performance on {task}")
        plt.tight_layout()
        plt.savefig(f"{task.replace('/', '-')}.png")


def format_output(df: pd.DataFrame, output_format: str) -> str:
    """Format the DataFrame output for markdown or LaTeX."""
    df = df.fillna("")
    if output_format == "markdown":
        return df.to_markdown(mode="github", floatfmt=".3g")
    elif output_format == "latex":
        return df.to_latex(float_format="%.3g")
    return ""


def main(
    result_dir: str,
    model_list: List[str],
    output_path: Optional[str] = None,
    output_format: str = "markdown",
    plot_bar: bool = False,
    plot_corr: bool = False,
    update_pages: bool = False,
):
    df = process_results(result_dir, model_list)
    if plot_corr:
        plot_correlation(df, "correlation.png")
    # plot_correlation(df.T, "correlation_model.png")
    if plot_bar:
        plot_task_performance(df)

    table = format_output(df, output_format)
    print(table)

    if output_path:
        with open(output_path, "w") as f:
            f.write(table)

    if update_pages:
        generate_json_path(df, "github_pages/public/leaderboard.json")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=str,
        default="result",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="leaderboard.md",
        help="Output path for the leaderboard",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="markdown",
        choices=["markdown", "latex"],
        help="Output format",
    )
    parser.add_argument(
        "--plot_bar", action="store_true", help="Plot bar plots for each task"
    )
    parser.add_argument(
        "--plot_corr", action="store_true", help="Plot correlation matrix between tasks"
    )
    parser.add_argument(
        "--update_pages", action="store_true", help="Update the GitHub Pages JSON"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_list = [
        "stabilityai/japanese-instructblip-alpha",
        "stabilityai/japanese-stable-vlm",
        "SakanaAI/Llama-3-EvoVLM-JP-v2",
        "cyberagent/llava-calm2-siglip",
        "llm-jp/llm-jp-3-vila-14b",
        "sbintuitions/sarashina2-vision-8b",
        "sbintuitions/sarashina2-vision-14b",
        "MIL-UT/Asagi-14B",
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "neulab/Pangea-7B-hf",
        "mistralai/Pixtral-12B-2409",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "Efficient-Large-Model/VILA1.5-13b",
        "OpenGVLab/InternVL2-8B",
        "OpenGVLab/InternVL2-26B",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "microsoft/Phi-4-multimodal-instruct",
        "gpt-4o-2024-11-20",
    ]
    main(
        args.result_dir,
        model_list,
        args.output_path,
        args.output_format,
        args.plot_bar,
        args.plot_corr,
        args.update_pages,
    )
