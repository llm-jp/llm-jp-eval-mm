import glob
import pandas as pd
import json
from datasets import Dataset, concatenate_datasets, load_dataset
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import click


@click.command()
@click.option("--task_id", type=str, default="japanese-heron-bench")
@click.option("--result_dir", type=str, default="results/run_0")
def main(task_id: str, result_dir: str):
    result_dir = f"{result_dir}/{task_id}"
    files = glob.glob(os.path.join(result_dir, "prediction/*.jsonl"))
    print(files)

    model_ids = ["-".join(pathlib.Path(file).stem.split("-")[:-1]) for file in files]
    print(model_ids)

    if task_id == "japanese-heron-bench":
        dataset = load_dataset("Silviase/Japanese-Heron-Bench", split="train")
        # question id, model_1's answer, model_2's answer, model_3's answer, ...
        df = pd.DataFrame()
        for example in dataset:
            if len(df) == 0:
                df["question_id"] = [example["question_id"]]
            if example["question_id"] not in df["question_id"].values:
                df = df._append(
                    {"question_id": example["question_id"]}, ignore_index=True
                )
            answer = example["answer"]
            answer_models = [
                "claude-3-opus-20240229",
                "gpt-4-0125-preview",
                "gpt-4-vision-preview",
            ]
            df.loc[df["question_id"] == example["question_id"], "category"] = example[
                "category"
            ]
            df.loc[df["question_id"] == example["question_id"], "context"] = example[
                "context"
            ]
            df.loc[df["question_id"] == example["question_id"], "input_text"] = example[
                "text"
            ]
            for model_id in answer_models:
                df.loc[df["question_id"] == example["question_id"], model_id] = answer[
                    model_id
                ]

        for file in files:
            with open(file, "r") as f:
                model_id = "-".join(pathlib.Path(file).stem.split("-")[:-1])
                for line in f:
                    data = json.loads(line)
                    if len(df) == 0:
                        df["question_id"] = [data["question_id"]]
                    if data["question_id"] not in df["question_id"].values:
                        df = df._append(
                            {"question_id": data["question_id"]}, ignore_index=True
                        )
                    df.loc[df["question_id"] == data["question_id"], model_id] = (
                        data["text"] + "\n" + "score: " + str(data["score"])
                    )
        print(df.head())
        df.to_excel(os.path.join(result_dir, "prediction.xlsx"), index=False)

        metrics_for_model = {}
        for file in files:
            with open(file, "r") as f:
                model_id = "-".join(pathlib.Path(file).stem.split("-")[:-1])
                metrics_for_model[model_id] = {
                    "detail": [],
                    "conv": [],
                    "complex": [],
                    "overall": [],
                }
                for example, line in zip(dataset, f):
                    data = json.loads(line)
                    metrics_for_model[model_id][example["category"]].append(
                        data["score"]
                    )
                    metrics_for_model[model_id]["overall"].append(data["score"])
        # プロット
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")

        # 各モデルのスコアをプロット
        colors = sns.color_palette("Set2", n_colors=len(metrics_for_model))
        # box plot
        df = pd.DataFrame()
        for model_id, metrics in metrics_for_model.items():
            for category, scores in metrics.items():
                for score in scores:
                    df = df._append(
                        {"model_id": model_id, "category": category, "score": score},
                        ignore_index=True,
                    )
        sns.boxplot(data=df, x="category", y="score", hue="model_id", palette=colors)

        # ラベルとタイトル
        plt.ylabel("Score")
        plt.xlabel("Category")
        # outer legend
        plt.legend(title="Models", loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "stripplot.png"))

        # result/evaluation/*.jsonl to excel
        files = glob.glob(os.path.join(result_dir, "evaluation/*.jsonl"))
        print(files)
        model_ids = [
            "-".join(pathlib.Path(file).stem.split("-")[:-1]) for file in files
        ]
        print(model_ids)
        # model_id, detail, conv, complex, overall
        df = pd.DataFrame()
        for i, file in enumerate(files):
            with open(file, "r") as f:
                model_id = "-".join(pathlib.Path(file).stem.split("-")[:-1])
                print(model_id)
                line = f.readline()
                data = json.loads(line)
                df.loc[i, "model_id"] = model_id
                df.loc[i, "detail"] = data["detail"]
                df.loc[i, "conv"] = data["conv"]
                df.loc[i, "complex"] = data["complex"]
                df.loc[i, "overall"] = data["overall"]
                df.loc[i, "parse_error_count"] = data["parse_error_count"]
                df.loc[i, "detail_rel"] = data["detail_rel"]
                df.loc[i, "conv_rel"] = data["conv_rel"]
                df.loc[i, "complex_rel"] = data["complex_rel"]
                df.loc[i, "overall_rel"] = data["overall_rel"]
                df.loc[i, "openai_model_id"] = data["openai_model_id"]

        # {"conv": 5.142857142857143, "conv_rel": 63.90532544378699, "detail": 5.333333333333333, "detail_rel": 67.87878787878788, "complex": 4.975, "complex_rel": 60.48632218844985, "parse_error_count": 1, "overall": 5.116504854368932, "overall_rel": 64.09014517034156, "openai_model_id": "gpt-4o-mini-2024-07-18"}

        print(df.head())

        df.to_excel(os.path.join(result_dir, "evaluation.xlsx"), index=False)
    elif task_id == "ja-vg-vqa-500":
        df = pd.DataFrame()
        for file in files:
            with open(file, "r") as f:
                model_id = "-".join(pathlib.Path(file).stem.split("-")[:-1])
                for line in f:
                    data = json.loads(line)
                    if len(df) == 0:
                        df["question_id"] = [data["question_id"]]
                    if data["question_id"] not in df["question_id"].values:
                        df = df._append(
                            {"question_id": data["question_id"]}, ignore_index=True
                        )

                    df.loc[df["question_id"] == data["question_id"], model_id] = (
                        data["text"]
                        + "\n"
                        + "score: "
                        + str(data["score_rougeL"])
                        + ", "
                        + str(data["score_llm_as_a_judge"])
                    )

        dataset = load_dataset("SakanaAI/JA-VG-VQA-500", split="test")

        # flatten "qas" to q/a/id triplets
        def flatten_sample(sample):
            dataset = {
                "image_id": [sample["image_id"] for _ in sample["qas"]],
                "image": [sample["image"] for _ in sample["qas"]],
                "qa_id": [qa["qa_id"] for qa in sample["qas"]],
                "question": [qa["question"] for qa in sample["qas"]],
                "answer": [qa["answer"] for qa in sample["qas"]],
            }
            return Dataset.from_dict(dataset)

        fragments = []
        for i, sample in enumerate(dataset):
            data_fragment = flatten_sample(sample)
            fragments.append(data_fragment)

        ds = concatenate_datasets(fragments)
        df["question_id"] = ds["qa_id"]
        df["question"] = ds["question"]
        df["answer"] = ds["answer"]
        print(df.head())
        df.to_excel(os.path.join(result_dir, "prediction.xlsx"), index=False)

        files = glob.glob(os.path.join(result_dir, "evaluation/*.jsonl"))
        print(files)
        model_ids = [
            "-".join(pathlib.Path(file).stem.split("-")[:-1]) for file in files
        ]
        print("model_id", model_ids)

        df = pd.DataFrame()
        for file in files:
            with open(file, "r") as f:
                model_id = "-".join(pathlib.Path(file).stem.split("-")[:-1])
                line = f.readline()
                data = json.loads(line)
                for key, value in data.items():
                    df.loc[model_id, key] = value

        print(df.head())
        df.to_excel(os.path.join(result_dir, "evaluation.xlsx"), index=True)

    elif task_id == "ja-vlm-bench-in-the-wild":
        df = pd.DataFrame()
        for file in files:
            with open(file, "r") as f:
                model_id = "-".join(pathlib.Path(file).stem.split("-")[:-1])
                for line in f:
                    data = json.loads(line)
                    if len(df) == 0:
                        df["question_id"] = [data["question_id"]]
                    if data["question_id"] not in df["question_id"].values:
                        df = df._append(
                            {"question_id": data["question_id"]}, ignore_index=True
                        )

                    df.loc[df["question_id"] == data["question_id"], model_id] = (
                        data["text"]
                        + "\n"
                        + "score: "
                        + str(data["score_rougeL"])
                        + ", "
                        + str(data["score_llm_as_a_judge"])
                    )

        ds = load_dataset("SakanaAI/JA-VLM-Bench-In-the-Wild", split="test")
        ds = ds.map(lambda example, idx: {"question_id": idx}, with_indices=True)
        df["question_id"] = ds["question_id"]
        df["question"] = ds["question"]
        df["answer"] = ds["answer"]
        print(df.head())
        df.to_excel(os.path.join(result_dir, "prediction.xlsx"), index=False)

        files = glob.glob(os.path.join(result_dir, "evaluation/*.jsonl"))
        print(files)
        model_ids = [
            "-".join(pathlib.Path(file).stem.split("-")[:-1]) for file in files
        ]
        print("model_id", model_ids)

        df = pd.DataFrame()
        for file in files:
            with open(file, "r") as f:
                model_id = pathlib.Path(file).stem
                line = f.readline()
                data = json.loads(line)
                for key, value in data.items():
                    df.loc[model_id, key] = value

        print(df.head())
        df.to_excel(os.path.join(result_dir, "evaluation.xlsx"), index=True)


if __name__ == "__main__":
    main()
