import json
import os
import pandas as pd
from argparse import ArgumentParser
from loguru import logger
import eval_mm
import eval_mm.metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

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
    "cc-ocr": "CC-OCR",
    "cvqa": "CVQA",
    "ai2d": "AI2D",
    "blink": "BLINK",
    "docvqa": "DocVQA",
    "infographicvqa": "InfoVQA",
    "textvqa": "TextVQA",
    "chartqa": "ChartQA",
    "chartqapro": "ChartQAPro",
    "mathvista": "MathVista",
    "okvqa": "OK-VQA",
}

TASK_CLUSTER_ALIAS = {
    "JMMMU": "言語・知識中心",
    "JDocQA": "言語・知識中心",
    "MECHA": "言語・知識中心",
    "JIC": "視覚中心",
    "VG-VQA": "視覚中心",
    "Heron": "視覚中心",
    "JVB-ItW": "視覚中心",
    "MulIm-VQA": "その他",
    "MMMU": "英語",
    "LLAVA": "英語",
    "CC-OCR": "言語・知識中心",
    "CVQA": "視覚中心",
    "AI2D": "英語",
    "BLINK": "英語",
    "DocVQA": "英語",
    "InfoVQA": "英語",
    "TextVQA": "英語",
    "ChartQA": "英語",
    "ChartQAPro": "英語",
    "MathVista": "英語",
    "OK-VQA": "英語",
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
    "cc-ocr": "macro_f1",
    "substring-match": "Acc",
    "cvqa": "Acc",
    "ai2d": "Acc",
    "blink": "Acc",
    "docvqa": "Acc",
    "infographicvqa": "Acc",
    "textvqa": "Acc",
    "chartqa": "Acc",
    "chartqapro": "Acc",
    "mathvista": "Acc",
    "okvqa": "Acc",
}

MODEL_LIST = [
    # "stabilityai/japanese-instructblip-alpha",
    "stabilityai/japanese-stable-vlm",
    "cyberagent/llava-calm2-siglip",
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "neulab/Pangea-7B-hf",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    # "OpenGVLab/InternVL2-8B",
    # "OpenGVLab/InternVL2-26B",
    "OpenGVLab/InternVL3-1B",
    "OpenGVLab/InternVL3-2B",
    "OpenGVLab/InternVL3-8B",
    "OpenGVLab/InternVL3-9B",
    "OpenGVLab/InternVL3-14B",
    "OpenGVLab/InternVL3-38B",
    "OpenGVLab/InternVL3-78B",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "gpt-4o-2024-11-20",
    # "mistralai/Pixtral-12B-2409",
    "llm-jp/llm-jp-3-vila-14b",
    # "Efficient-Large-Model/VILA1.5-13b",
    "SakanaAI/Llama-3-EvoVLM-JP-v2",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "google/gemma-3-4b-pt",
    "google/gemma-3-12b-pt",
    "google/gemma-3-27b-pt",
    # "tokyotech-llm/gemma3_4b_exp8-checkpoint-50000",
    "sbintuitions/sarashina2-vision-8b",
    "sbintuitions/sarashina2-vision-14b",
    "microsoft/Phi-4-multimodal-instruct",
    "MIL-UT/Asagi-14B",
    "turing-motors/Heron-NVILA-Lite-1B",
    "turing-motors/Heron-NVILA-Lite-2B",
    "turing-motors/Heron-NVILA-Lite-15B",
    "turing-motors/Heron-NVILA-Lite-33B",
    "CohereLabs/aya-vision-8b",
    "CohereLabs/aya-vision-32b",
]


def load_evaluation_data(result_dir: str, model: str, task_dirs: list[str]) -> dict:
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
            if metric not in eval_mm.ScorerRegistry.get_metric_list():
                logger.warning(f"Skipping unsupported metric: {metric}")
                continue
            overall_score = aggregate_output["overall_score"]
            if metric in [
                "jdocqa",
                "jmmmu",
                "jic-vqa",
                "mecha-ja",
                "mmmu",
                "cc-ocr",
                "substring-match",
                "ai2d",
                "blink",
                "mathvista",
            ]:
                overall_score = overall_score * 100
            model_results[f"{task_dir}/{metric}"] = overall_score

    return model_results


def process_results(
    result_dir: str,
    model_list: list[str],
    add_avg: bool = False,
    task_id_list: list[str] | None = None,
) -> pd.DataFrame:
    """Process all evaluation results into a structured DataFrame."""
    if task_id_list:
        task_dirs = task_id_list
    else:
        task_dirs = [d for d in os.listdir(result_dir) if not d.startswith(".")]

    df = pd.DataFrame()

    for model in model_list:
        logger.info(f"Processing results for {model}")
        model_results = load_evaluation_data(result_dir, model, task_dirs)
        if len(model_results) == 1:
            continue
        df = df._append(model_results, ignore_index=True)

    df = df.set_index("Model").round(2)
    df = df.rename(
        columns={
            k: f"{TASK_ALIAS[k.split('/')[0]]}/{METRIC_ALIAS[k.split('/')[1]]}"
            for k in df.columns
        }
    )
    if add_avg:
        # すべてのスコアを 100 点満点に正規化
        df_normalized = df.apply(lambda x: x / x.max() * 100, axis=0)

        # 各モデルの全体スコア（平均）を計算し、最後の列に追加
        df["Avg/Avg"] = df_normalized.mean(axis=1).round(2)

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
                if "/" not in col:
                    model_entry["scores"][col] = score
                    continue
                task, metric = col.split("/")
                if task not in model_entry["scores"]:
                    model_entry["scores"][task] = {}
                model_entry["scores"][task][metric] = score

        json_data.append(model_entry)

    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"JSON output saved to {output_path}")


def plot_correlation(df: pd.DataFrame, filename: str, tex_columns: list[str] = None):
    """Plot and save the correlation matrix."""
    
    # Filter columns if tex_columns is provided
    if tex_columns:
        df = df[tex_columns]
    
    # Rename columns to show only task names
    rename_dict = {}
    for col in df.columns:
        task_name = col.split('/')[0] if '/' in col else col
        rename_dict[col] = task_name
    df = df.rename(columns=rename_dict)
    
    # Define task order as in TeX output: English tasks → Japanese tasks
    task_order = [
        # English tasks (from en_task_order)
        "OK-VQA", "TextVQA", "AI2D", "ChartQA", "DocVQA", "BLINK", "InfoVQA", "MMMU", "LLAVA",
        # Japanese tasks (from ja_task_order)
        "CVQA", "CC-OCR", "JIC", "MulIm-VQA", "JMMMU", "JDocQA", "JVB-ItW", "VG-VQA", "Heron", "MECHA"
    ]
    
    # Filter and reorder dataframe columns based on task_order
    available_tasks = [task for task in task_order if task in df.columns]
    df = df[available_tasks]
    
    corr = df.corr(method="spearman")
    
    # Create a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap with lower triangle only
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",  # Red (high) to Blue (low) colormap
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10, "weight": "bold"},
        ax=ax,
        vmin=0,  # Set minimum to 0
        vmax=1,
    )
    
    # Make tick labels bold and larger
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=11, fontweight='bold', rotation=45, ha='right')
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=11, fontweight='bold', rotation=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')


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


def calculate_average_correlations(df: pd.DataFrame, tex_columns: list[str] = None):
    """Calculate average Spearman correlations within and between task groups."""
    # Filter columns if tex_columns is provided
    if tex_columns:
        df = df[tex_columns]
    
    # Rename columns to show only task names
    rename_dict = {}
    for col in df.columns:
        task_name = col.split('/')[0] if '/' in col else col
        rename_dict[col] = task_name
    df = df.rename(columns=rename_dict)
    
    # Define English and Japanese tasks
    english_tasks = ["OK-VQA", "TextVQA", "AI2D", "ChartQA", "DocVQA", "BLINK", "InfoVQA", "MMMU", "LLAVA"]
    japanese_tasks = ["CVQA", "CC-OCR", "JIC", "MulIm-VQA", "JMMMU", "JDocQA", "JVB-ItW", "VG-VQA", "Heron", "MECHA"]
    
    # Filter available tasks
    available_english = [task for task in english_tasks if task in df.columns]
    available_japanese = [task for task in japanese_tasks if task in df.columns]
    
    # Calculate correlation matrix
    corr = df.corr(method="spearman")
    
    # Calculate average correlations within English tasks
    english_corr_values = []
    for i, task1 in enumerate(available_english):
        for j, task2 in enumerate(available_english):
            if i < j:  # Only upper triangle (avoid duplicates and diagonal)
                english_corr_values.append(corr.loc[task1, task2])
    avg_english_corr = np.mean(english_corr_values) if english_corr_values else 0
    
    # Calculate average correlations within Japanese tasks
    japanese_corr_values = []
    for i, task1 in enumerate(available_japanese):
        for j, task2 in enumerate(available_japanese):
            if i < j:  # Only upper triangle
                japanese_corr_values.append(corr.loc[task1, task2])
    avg_japanese_corr = np.mean(japanese_corr_values) if japanese_corr_values else 0
    
    # Calculate average correlations between English and Japanese tasks
    cross_corr_values = []
    for en_task in available_english:
        for ja_task in available_japanese:
            cross_corr_values.append(corr.loc[en_task, ja_task])
    avg_cross_corr = np.mean(cross_corr_values) if cross_corr_values else 0
    
    # Print results
    print("\n" + "="*60)
    print("Average Spearman Correlations:")
    print("="*60)
    print(f"Within English tasks:   {avg_english_corr:.3f}")
    print(f"  - Based on {len(english_corr_values)} pairs from {len(available_english)} tasks")
    print(f"Within Japanese tasks:  {avg_japanese_corr:.3f}")
    print(f"  - Based on {len(japanese_corr_values)} pairs from {len(available_japanese)} tasks")
    print(f"Between EN-JA tasks:    {avg_cross_corr:.3f}")
    print(f"  - Based on {len(cross_corr_values)} pairs")
    print("="*60 + "\n")
    
    return {
        "english_internal": avg_english_corr,
        "japanese_internal": avg_japanese_corr,
        "english_japanese_cross": avg_cross_corr
    }


def plot_task_clustering(df: pd.DataFrame, filename: str, tex_columns: list[str] = None):
    """Plot hierarchical clustering dendrogram based on task correlations."""
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform
    
    # Filter columns if tex_columns is provided
    if tex_columns:
        df = df[tex_columns]
    
    # Rename columns to show only task names
    rename_dict = {}
    for col in df.columns:
        task_name = col.split('/')[0] if '/' in col else col
        rename_dict[col] = task_name
    df = df.rename(columns=rename_dict)
    
    # Define task order as in TeX output: English tasks → Japanese tasks
    task_order = [
        # English tasks
        "OK-VQA", "TextVQA", "AI2D", "ChartQA", "DocVQA", "BLINK", "InfoVQA", "MMMU", "LLAVA",
        # Japanese tasks
        "CVQA", "CC-OCR", "JIC", "MulIm-VQA", "JMMMU", "JDocQA", "JVB-ItW", "VG-VQA", "Heron", "MECHA"
    ]
    
    # Filter and reorder dataframe columns based on task_order
    available_tasks = [task for task in task_order if task in df.columns]
    df = df[available_tasks]
    
    # Calculate correlation matrix
    corr = df.corr(method="spearman")
    
    # Convert correlation to distance (1 - correlation)
    # This ensures high correlation = small distance
    distance_matrix = 1 - corr
    
    # Convert to condensed distance matrix
    condensed_distance = squareform(distance_matrix)
    
    # Perform hierarchical clustering
    linkage = sch.linkage(condensed_distance, method='average')
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot dendrogram
    dendrogram = sch.dendrogram(
        linkage,
        labels=corr.index.tolist(),
        orientation='top',
        color_threshold=0.7,  # Color threshold for clusters
        above_threshold_color='gray'
    )
    
    plt.title('Task Clustering based on Correlation Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Tasks', fontsize=12, fontweight='bold')
    plt.ylabel('Distance (1 - Spearman Correlation)', fontsize=12, fontweight='bold')
    
    # Make x-axis labels bold
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=11, fontweight='bold', rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create a horizontal dendrogram for better readability
    plt.figure(figsize=(10, 12))
    dendrogram_h = sch.dendrogram(
        linkage,
        labels=corr.index.tolist(),
        orientation='left',
        color_threshold=0.7,
        above_threshold_color='gray'
    )
    
    plt.title('Task Clustering based on Correlation Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Distance (1 - Spearman Correlation)', fontsize=12, fontweight='bold')
    plt.ylabel('Tasks', fontsize=12, fontweight='bold')
    
    # Make y-axis labels bold
    ax = plt.gca()
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    # Save horizontal version
    base_name = filename.rsplit('.', 1)[0]
    ext = filename.rsplit('.', 1)[1] if '.' in filename else 'png'
    plt.savefig(f"{base_name}_horizontal.{ext}", dpi=150, bbox_inches='tight')
    plt.close()


def compute_cluster_scores(df: pd.DataFrame) -> pd.DataFrame:
    df_filtered = df.loc[:, ~df.columns.str.contains("Rouge", case=False)]
    task_names = [col.split("/")[0] for col in df_filtered.columns]
    cluster_labels = [TASK_CLUSTER_ALIAS.get(task, None) for task in task_names]
    cluster_scores: dict[str, list[str]] = {
        c: [] for c in set(TASK_CLUSTER_ALIAS.values())
    }
    for col, label in zip(df_filtered.columns, cluster_labels):
        if label:
            cluster_scores[label].append(col)

    df_z = df_filtered.copy()
    df_z = (df_z - df_z.mean()) / df_z.std()

    cluster_z = {}
    for cluster, cols in cluster_scores.items():
        cluster_z[cluster] = df_z[cols].mean(axis=1)

    for cluster, z in cluster_z.items():
        df[f"{cluster}偏差値"] = (z * 10 + 50).round(1)

    overall_z = pd.concat(cluster_z.values(), axis=1).mean(axis=1)
    df["総合偏差値"] = (overall_z * 10 + 50).round(1)

    col_order = (
        ["総合偏差値"]
        + [f"{c}偏差値" for c in cluster_scores.keys()]
        + df_filtered.columns.tolist()
    )
    df = df.loc[:, col_order]

    return df


def format_output(df: pd.DataFrame, output_format: str) -> str:
    """Format the DataFrame output for markdown or LaTeX."""

    # textbf top1 score and underline top2 score for each task
    for col in df.columns:
        top1_model = df[col].astype(float).idxmax()
        top2_model = df[col].astype(float).nlargest(2).index[-1]
        top1_score = f"{float(df.loc[top1_model, col]):.1f}"
        top2_score = f"{float(df.loc[top2_model, col]):.1f}"
        # apply formatting
        if output_format == "latex":
            df.loc[top1_model, col] = f"\\textbf{{{top1_score}}}"
            df.loc[top2_model, col] = f"\\textit{{{top2_score}}}"
            df.loc[top2_model, col] = f"\\underline{{{top2_score}}}"
        else:
            df.loc[top1_model, col] = f"**{top1_score}**"
            df.loc[top2_model, col] = f"*{top2_score}*"
            df.loc[top2_model, col] = f"<u>{top2_score}</u>"

    df = df.fillna("")

    if output_format == "markdown":
        return df.to_markdown(mode="github", floatfmt=".1f")
    elif output_format == "latex":
        return df.to_latex(
            float_format="%.1f", column_format="l" + "c" * len(df.columns)
        )
    return ""


def export_to_tex_files(df: pd.DataFrame, ja_tasks: list[str], en_tasks: list[str]):
    """Export leaderboard results to LaTeX table files like artifact/result_*.tex."""
    
    # Define desired task order for Japanese tasks
    ja_task_order = ["CVQA", "CC-OCR", "JIC", "MulIm-VQA", "JMMMU", "JDocQA", "JVB-ItW", "VG-VQA", "Heron", "MECHA"]
    
    # Define desired task order for English tasks
    en_task_order = ["OK-VQA", "TextVQA", "AI2D", "ChartQA", "DocVQA", "BLINK", "InfoVQA", "MMMU", "LLAVA"]
    
    # Prepare data for Japanese tasks with specific ordering
    ja_columns_ordered = []
    for task in ja_task_order:
        for col in df.columns:
            if task in col:
                ja_columns_ordered.append(col)
                break
    ja_df = df[ja_columns_ordered].copy() if ja_columns_ordered else pd.DataFrame()
    
    # Prepare data for English tasks with specific ordering
    en_columns_ordered = []
    for task in en_task_order:
        for col in df.columns:
            if task in col:
                en_columns_ordered.append(col)
                break
    en_df = df[en_columns_ordered].copy() if en_columns_ordered else pd.DataFrame()
    
    # Export Japanese results
    if not ja_df.empty:
        export_single_tex(ja_df, "artifact/result_ja.tex", "日本語タスク", ja_task_order)
    
    # Export English results
    if not en_df.empty:
        export_single_tex(en_df, "artifact/result_en.tex", "英語タスク", en_task_order)
    
    # Return combined columns for correlation matrix
    return ja_columns_ordered + en_columns_ordered


def export_single_tex(df: pd.DataFrame, output_path: str, task_type: str, task_order: list[str] = None):
    """Export a DataFrame to a single LaTeX table file."""
    # Create artifact directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # First, identify top scores for each column
    top_scores = {}
    for col in df.columns:
        col_scores = df[col].dropna()
        if len(col_scores) > 0:
            top_scores[col] = {
                'top1': col_scores.max(),
                'top2': col_scores.nlargest(2).iloc[-1] if len(col_scores) > 1 else None
            }
    
    # Build LaTeX table header
    tex_content = []
    tex_content.append("\\begingroup")
    tex_content.append("\\setlength{\\tabcolsep}{2pt}")
    tex_content.append("\\renewcommand{\\arraystretch}{0.9}")
    tex_content.append("\\begin{table*}[t]")
    tex_content.append("\\centering")
    tex_content.append("\\footnotesize")
    tex_content.append(f"\\caption{{競争力のある視覚言語モデルの\\methodName を用いた\\textbf{{{task_type}}}での評価例．")
    tex_content.append("``--''は評価データセットを学習に用いているためスコアが算出できないことを示す．\\textbf{太字}は最も高い結果．\\underline{下線}は二番目に高い結果を示している．}")
    tex_content.append("\\vspace{.2em}")
    tex_content.append("\\begin{adjustbox}{max width=\\linewidth}")
    
    # Determine column alignment
    col_align = "l" + "c" * len(df.columns)
    tex_content.append(f"\\begin{{tabular}}{{{col_align}}}")
    tex_content.append("\\toprule")
    
    # Build header row with task names
    header_parts = ["\\multirow{2}{*}{\\textbf{Models}}"]
    for col in df.columns:
        task_name = col.split('/')[0]
        header_parts.append(f"\\textbf{{{task_name}}}")
    tex_content.append(" & ".join(header_parts) + "\\\\")
    
    # Add cmidrule for each column
    cmidrules = []
    for i in range(2, len(df.columns) + 2):
        cmidrules.append(f"\\cmidrule(lr){{{i}-{i}}}")
    tex_content.append("".join(cmidrules))
    
    # Add metric row
    metric_parts = [""]
    for col in df.columns:
        metric = col.split('/')[1] if '/' in col else "Acc."
        task = col.split('/')[0] if '/' in col else ""
        
        if metric == "macro_f1":
            metric = "F1"
        elif metric == "LLM":
            # Heron task shows "LLM(%)" in Japanese table
            if task == "Heron":
                metric = "LLM(%)"
            else:
                metric = "LLM"
        else:
            metric = "Acc."
        metric_parts.append(metric)
    tex_content.append(" & ".join(metric_parts) + " \\\\")
    tex_content.append("\\midrule")
    
    # Group models by origin with specific ordering
    # Define model groups and their display names
    domestic_model_mapping = {
        "cyberagent/llava-calm2-siglip": "Llava-calm2-siglip",
        "SakanaAI/Llama-3-EvoVLM-JP-v2": "Llama-3-EvoVLM-JP-v2",
        "MIL-UT/Asagi-14B": "Asagi-14B",
        "sbintuitions/sarashina2-vision-8b": "Sarashina2-Vision-8b",
        "sbintuitions/sarashina2-vision-14b": "Sarashina2-Vision-14b",
        "turing-motors/Heron-NVILA-Lite-1B": "Heron-NVILA-Lite-1B",
        "turing-motors/Heron-NVILA-Lite-2B": "Heron-NVILA-Lite-2B",
        "turing-motors/Heron-NVILA-Lite-15B": "Heron-NVILA-Lite-15B",
        "turing-motors/Heron-NVILA-Lite-33B": "Heron-NVILA-Lite-33B",
        "llm-jp/llm-jp-3-vila-14b": "llm-jp-3-vila-14b"
    }
    
    # Desired order for domestic models
    domestic_order = [
        "cyberagent/llava-calm2-siglip", 
        "SakanaAI/Llama-3-EvoVLM-JP-v2",
        "MIL-UT/Asagi-14B",
        "sbintuitions/sarashina2-vision-8b",
        "sbintuitions/sarashina2-vision-14b",
        "turing-motors/Heron-NVILA-Lite-1B",
        "turing-motors/Heron-NVILA-Lite-2B",
        "turing-motors/Heron-NVILA-Lite-15B",
        "turing-motors/Heron-NVILA-Lite-33B",
        "llm-jp/llm-jp-3-vila-14b"
    ]
    
    # Use multicolumn spanning with proper column count
    num_cols = len(df.columns) + 1  # +1 for model name column
    
    # Add domestic models section
    tex_content.append(f"\\multicolumn{{{num_cols}}}{{l}}{{\\textbf{{日本国内で開発されたモデル}}}} \\\\")
    tex_content.append("\\midrule")
    
    # Add domestic models in specific order
    for model_full in domestic_order:
        if model_full in df.index:
            model_display = domestic_model_mapping.get(model_full, model_full.split('/')[-1])
            row = format_model_row_with_highlights(model_display, df.loc[model_full], top_scores)
            tex_content.append(row)
    
    # Add foreign models section
    tex_content.append("\\midrule")
    tex_content.append(f"\\multicolumn{{{num_cols}}}{{l}}{{\\textbf{{海外で開発されたモデル}}}} \\\\")
    tex_content.append("\\midrule")
    
    # Collect foreign models
    foreign_models = []
    for model in df.index:
        if model not in domestic_model_mapping:
            foreign_models.append(model)
    
    # Sort foreign models alphabetically by their display name
    foreign_models.sort(key=lambda x: x.split('/')[-1])
    
    # Define specific foreign model display names
    foreign_model_mapping = {
        "llava-hf/llava-1.5-7b-hf": "Llava-1.5-7b",
        "llava-hf/llava-v1.6-mistral-7b-hf": "Llava-v1.6-mistral-7b",
        "CohereLabs/aya-vision-8b": "Aya-Vision-8b",
        "CohereLabs/aya-vision-32b": "Aya-Vision-32b",
        "neulab/Pangea-7B-hf": "Pangea-7B",
        "microsoft/Phi-4-multimodal-instruct": "Phi-4-multimodal-instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct": "Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct": "Llama-3.2-90B-Vision-Instruct",
        "google/gemma-3-4b-it": "Gemma-3-4b-it",
        "google/gemma-3-12b-it": "Gemma-3-12b-it",
        "google/gemma-3-27b-it": "Gemma-3-27b-it",
        "OpenGVLab/InternVL3-1B": "InternVL3-1B",
        "OpenGVLab/InternVL3-2B": "InternVL3-2B",
        "OpenGVLab/InternVL3-8B": "InternVL3-8B",
        "OpenGVLab/InternVL3-14B": "InternVL3-14B",
        "OpenGVLab/InternVL3-38B": "InternVL3-38B",
        "OpenGVLab/InternVL3-78B": "InternVL3-78B",
        "Qwen/Qwen2-VL-7B-Instruct": "Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2-VL-72B-Instruct": "Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B-Instruct",
    }
    
    # Define desired foreign model order
    foreign_model_order = [
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "CohereLabs/aya-vision-8b",
        "CohereLabs/aya-vision-32b",
        "neulab/Pangea-7B-hf",
        "microsoft/Phi-4-multimodal-instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "OpenGVLab/InternVL3-1B",
        "OpenGVLab/InternVL3-2B",
        "OpenGVLab/InternVL3-8B",
        "OpenGVLab/InternVL3-14B",
        "OpenGVLab/InternVL3-38B",
        "OpenGVLab/InternVL3-78B",
        "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
    ]
    
    # Use the specified order for foreign models
    for model in foreign_model_order:
        if model in df.index:
            model_display = foreign_model_mapping.get(model, model.split('/')[-1])
            row = format_model_row_with_highlights(model_display, df.loc[model], top_scores)
            tex_content.append(row)
    
    # Close table
    tex_content.append("\\bottomrule")
    tex_content.append("\\end{tabular}")
    tex_content.append("\\end{adjustbox}")
    tex_content.append("\\end{table*}")
    tex_content.append("\\endgroup")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_content))
    
    logger.info(f"Exported results to {output_path}")


def format_model_row_with_highlights(model_name: str, scores: pd.Series, top_scores: dict) -> str:
    """Format a single model row for LaTeX table with top score highlighting."""
    # Use model_name as is (already processed)
    model_display = model_name
    
    parts = [model_display]
    
    for col in scores.index:
        score = scores[col]
        if pd.isna(score):
            parts.append("--")
        else:
            # Format score based on metric type
            if isinstance(score, (int, float)):
                # Special handling for LLAVA scores (shown without decimal)
                if "LLAVA" in col:
                    formatted_score = f"{score:.1f}"
                elif score < 1:  # Likely a percentage in decimal form
                    formatted_score = f"{score * 100:.1f}"
                else:
                    formatted_score = f"{score:.1f}"
            else:
                formatted_score = str(score)
            
            # Apply formatting for top scores
            if col in top_scores:
                if score == top_scores[col]['top1']:
                    formatted_score = f"\\textbf{{{formatted_score}}}"
                elif top_scores[col]['top2'] is not None and score == top_scores[col]['top2']:
                    formatted_score = f"\\underline{{{formatted_score}}}"
            
            parts.append(formatted_score)
    
    return " & ".join(parts) + " \\\\"




def main(
    result_dir: str,
    model_list: list[str],
    output_path: str | None = None,
    output_format: str = "markdown",
    plot_bar: bool = False,
    plot_corr: bool = False,
    update_pages: bool = False,
    add_avg: bool = False,
    task_id_list: list[str] | None = None,
    add_clusterscore: bool = False,
    export_tex: bool = False,
):
    df = process_results(result_dir, model_list, add_avg, task_id_list)
    
    if add_clusterscore:
        df = compute_cluster_scores(df)

    if plot_bar:
        plot_task_performance(df.copy())

    table = format_output(df.copy(), output_format)
    print(table)

    if output_path:
        with open(output_path, "w") as f:
            f.write(table)

    if update_pages:
        # sort by task
        df = df.sort_index(axis=1)
        generate_json_path(df.copy(), "github_pages/public/leaderboard.json")
    
    tex_columns = None
    if export_tex:
        # Define Japanese and English tasks based on TASK_CLUSTER_ALIAS
        ja_tasks = [task for task, cluster in TASK_CLUSTER_ALIAS.items() 
                   if cluster in ["言語・知識中心", "視覚中心", "その他"]]
        en_tasks = [task for task, cluster in TASK_CLUSTER_ALIAS.items() 
                   if cluster == "英語"]
        
        tex_columns = export_to_tex_files(df.copy(), ja_tasks, en_tasks)
    
    if plot_corr:
        plot_correlation(df.copy(), "correlation.png", tex_columns)
        # Also plot clustering dendrogram
        plot_task_clustering(df.copy(), "task_clustering.png", tex_columns)
        # Calculate and print average correlations
        calculate_average_correlations(df.copy(), tex_columns)


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
    parser.add_argument(
        "--add_avg", action="store_true", help="Add average score column"
    )
    parser.add_argument(
        "--add_clusterscore",
        action="store_true",
        help="Add category-based and overall deviation scores (偏差値)",
    )
    parser.add_argument(
        "--task_id_list",
        type=str,
        default=None,
        nargs="+",
        help=f"List of task IDs to include in the leaderboard. Available: {TASK_ALIAS.keys()}",
    )
    parser.add_argument(
        "--export_tex",
        action="store_true",
        help="Export leaderboard results to artifact/result_*.tex files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(args.task_id_list)
    main(
        args.result_dir,
        MODEL_LIST,
        args.output_path,
        args.output_format,
        args.plot_bar,
        args.plot_corr,
        args.update_pages,
        args.add_avg,
        args.task_id_list,
        args.add_clusterscore,
        args.export_tex,
    )
