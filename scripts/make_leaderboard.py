import json
import os
import pandas as pd
from argparse import ArgumentParser
from typing import Dict, List, Optional

BENCHMARK_METRICS = {
    "japanese-heron-bench": [
        "conv",
        "detail",
        "complex",
        "overall",
        "conv_rel",
        "detail_rel",
        "complex_rel",
        "overall_rel",
    ],
    "ja-vlm-bench-in-the-wild": [
        "rougel",
        "llm_as_a_judge",
    ],
    "ja-vg-vqa-500": [
        "rougel",
        "llm_as_a_judge",
    ],
    "jdocqa": [
        "yesno_exact",
        "factoid_exact",
        "numerical_exact",
        "open-ended_bleu",
        "overall",
    ],
    "ja-multi-image-vqa": [
        "rougel",
        "llm_as_a_judge",
    ],
    "jmmmu": [
        "Overall-Art and Psychology",
        "Design",
        "Music",
        "Psychology",
        "Overall-Business",
        "Accounting",
        "Economics",
        "Finance",
        "Manage",
        "Marketing",
        "Overall-Science",
        "Biology",
        "Chemistry",
        "Math",
        "Physics",
        "Overall-Health and Medicine",
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
        "Overall-Tech and Engineering",
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
        "Overall",
    ],
    "jic_vqa": ["jafacility20", "jaflower30", "jafood101", "jalandmark10", "average"],
    "mecha-ja": [
        "overall",
        "Factoid",
        "Non-Factoid",
        "with_background",
        "without_background",
    ],
    "llava-bench-in-the-wild": [
        "rougel",
        "llm_as_a_judge",
    ],
    "mmmu": [
        "Overall-Art and Design",
        "Art",
        "Art_Theory",
        "Design",
        "Music",
        "Overall-Business",
        "Accounting",
        "Economics",
        "Finance",
        "Manage",
        "Marketing",
        "Overall-Science",
        "Biology",
        "Chemistry",
        "Geography",
        "Math",
        "Physics",
        "Overall-Health and Medicine",
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
        "Overall-Humanities and Social Science",
        "History",
        "Literature",
        "Sociology",
        "Psychology",
        "Overall-Tech and Engineering",
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
        "Overall",
    ],
}

MAIN_METRICS = {
    "japanese-heron-bench": ["llm_as_a_judge_heron_bench-overall_rel"],
    "ja-vlm-bench-in-the-wild": ["llm_as_a_judge", "rougel"],
    "ja-vg-vqa-500": ["llm_as_a_judge"],
    "jdocqa": ["jdocqa-overall"],
    "ja-multi-image-vqa": ["rougel"],
    "jmmmu": ["jmmmu-Overall"],
    "jic-vqa": ["jic_vqa-average"],
    "mecha-ja": ["mecha-ja-overall"],
    "llava-bench-in-the-wild": ["llm_as_a_judge"],
    "mmmu": ["mmmu-Overall"],
}

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
    "llm_as_a_judge_heron_bench-overall_rel": "LLM",
    "llm_as_a_judge": "LLM",
    "rougel": "Rouge",
    "jdocqa-overall": "Acc",
    "jmmmu-Overall": "Acc",
    "jic_vqa-average": "Acc",
    "mecha-ja-overall": "Acc",
    "mmmu-Overall": "Acc",
}


def read_evaluation_jsonl(file_path: str) -> Dict:
    merged_result = {}
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            merged_result.update(data)
    return merged_result


def main(result_dir: str, model_list: List[str], output_path: Optional[str] = None):
    task_dirs = [d for d in os.listdir(result_dir) if not d.startswith(".")]

    print(f"Tasks found: {task_dirs}")
    df = pd.DataFrame()

    for model in model_list:
        model_results = {"Model": model}

        for task_dir in task_dirs:
            eval_path = os.path.join(result_dir, task_dir, model, "evaluation.json")
            if not os.path.exists(eval_path):
                print(f"Warning: {eval_path} not found.")
                continue

            evaluation = read_evaluation_jsonl(eval_path)

            # ネストされている形式にも対応
            for key, val in evaluation.items():
                if isinstance(val, dict):
                    for sub_metric, sub_val in val.items():
                        if sub_metric == "parse_error_count":
                            continue  # スキップしたい項目
                        model_results[f"{task_dir}-{key}-{sub_metric}"] = sub_val
                else:
                    model_results[f"{task_dir}-{key}"] = val

        df = pd.concat([df, pd.DataFrame([model_results])], ignore_index=True)

    # 全体結果出力
    print("\n## Full Benchmark Result")
    print(df.to_markdown(mode="github", index=False))

    # メインのメトリクスだけを抽出
    main_df = df[["Model"]]
    for task, metrics in MAIN_METRICS.items():
        for metric in metrics:
            print(f"{task}-{metric}")
            if f"{task}-{metric}" in df.columns:
                print(f"{metric}: {df[f'{task}-{metric}'].mean()}")
                main_df[f"{TASK_ALIAS[task]}-{METRIC_ALIAS[metric]}"] = df[
                    f"{task}-{metric}"
                ]
            else:
                main_df[f"{TASK_ALIAS[task]}-{METRIC_ALIAS[metric]}"] = None

    print("\n## Main Metrics")
    print(main_df.to_markdown(mode="github", index=False))

    if output_path:
        with open(output_path, "w") as f:
            f.write(main_df.to_markdown(mode="github", index=False))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="paper")
    parser.add_argument("--output_path", type=str, default="leaderboard.md")
    args = parser.parse_args()

    # モデルは実行時引数でも取れるようにしても良い
    model_list = [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "sbintuitions/sarashina2-vision-8b",
        "sbintuitions/sarashina2-vision-14b",
        "google/gemma-3-12b-it",
    ]

    main(args.result_dir, model_list, args.output_path)
