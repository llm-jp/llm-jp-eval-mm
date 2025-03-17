import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa
import numpy as np

# ======================================
# モデルごとのjsonlファイルパスを設定
# ======================================
vis_dir = "logs/mecha-ja/visualize/"
root_dir = "logs/mecha-ja/prediction/"
files = os.listdir(root_dir)
model_names = [
    os.path.basename(f).replace(".jsonl", "") for f in files if f.endswith(".jsonl")
]

model_files = {
    model_name: os.path.join(root_dir, f"{model_name}.jsonl")
    for model_name in model_names
}

model_dfs = {}


# ======================================
# JSONLを読み込み、DataFrame化する関数
# ======================================
def load_jsonl_to_df(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    return pd.DataFrame(data_list)


# ======================================
# データ読み込み & 予測列の追加
# ======================================
def check_abcd(text):
    letters = ["A", "B", "C", "D"]
    found = [
        ch for ch in letters if ch in text
    ]  # テキスト中に含まれる A/B/C/D をリスト化
    # 含まれている文字がちょうど1つならその文字、そうでなければ F を返す
    return found[0] if len(found) == 1 else "F"


for model_name, file_path in model_files.items():
    df = load_jsonl_to_df(file_path)
    df["pred"] = df["text"].apply(check_abcd)
    model_dfs[model_name] = df

rotate_map = {
    "A": ["A", "D", "C", "B"],
    "B": ["B", "A", "D", "C"],
    "C": ["C", "B", "A", "D"],
    "D": ["D", "C", "B", "A"],
}

# ======================================
# (1) モデルごとの回答選択肢の分布を可視化（相対頻度）
# ======================================
# 3 x 3 のグリッドで可視化
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10), sharex=True, sharey=True)

# axesを1次元にする
axes = axes.flatten()

for ax, (model_name, df) in zip(axes, model_dfs.items()):
    # 予測回答の分布（相対頻度）をカウント
    pred_counts = (
        df["pred"].value_counts().reindex(["A", "B", "C", "D", "F"], fill_value=0)
    )
    pred_counts = pred_counts / pred_counts.sum()  # 相対頻度に変換

    ax.bar(
        pred_counts.index,
        pred_counts.values,
        color=["#FF9999", "#FFE888", "#99FF99", "#99CCFF", "#CCCCCC"],
    )

    # 0.25 に赤い線を引く
    ax.axhline(y=0.25, color="r", linestyle="--", linewidth=1)

    ax.set_title(f"{model_name}")
    ax.set_xlabel("選択肢")
    ax.set_ylabel("選択頻度")

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "prediction_distribution.png"))
plt.close()

# ======================================
# (2) soft accuracy, strict accuracy, consistency の計算
# --------------------------------------
# ・soft accuracy: 単純に df["mecha-ja"] の平均値
# ・strict accuracy: 同じ問題 (q_base) で全て mecha-ja==1 なら正解とカウント
# ・consistency: rot0 の予測に応じて rotate_map 通りになっている割合
# ======================================
results = []

for model_name, df in model_dfs.items():
    # soft accuracy
    soft_accuracy = df["mecha-ja"].mean()  # 1と0の平均 = 正解率

    # "X_rotY" の "X" 部分を q_base として抽出
    df["q_base"] = df["question_id"].apply(lambda x: x.split("_rot")[0])
    # 回転番号を取得
    df["rot"] = df["question_id"].apply(lambda x: int(x.split("_rot")[1]))

    grouped = df.groupby("q_base")
    unique_questions = df["q_base"].unique()

    correct_count = 0  # strict用
    consistent_count = 0

    for q_id, group in grouped:
        # strict正答率 (全rotで mecha-ja == 1)
        if all(group["mecha-ja"] == 1):
            correct_count += 1

        # 一貫性 (rotate_map に従っているか)
        group_sorted = group.sort_values("rot")
        preds = group_sorted["pred"].tolist()  # rot0→rot1→rot2→rot3 の順

        pred_rot0 = preds[0]  # 最初が rot0
        if pred_rot0 in rotate_map:
            expected_sequence = rotate_map[pred_rot0]
            # 一貫しているかどうか
            if len(preds) == 4 and preds == expected_sequence:
                consistent_count += 1

    strict_accuracy = (
        correct_count / len(unique_questions) if len(unique_questions) else 0
    )
    consistency = (
        consistent_count / len(unique_questions) if len(unique_questions) else 0
    )

    results.append(
        {
            "model": model_name,
            "soft_accuracy": soft_accuracy,
            "strict_accuracy": strict_accuracy,
            "consistency": consistency,
        }
    )

results_df = pd.DataFrame(results)
print(results_df)

# ======================================
# (3) 3種の指標 (soft, strict, consistency) を棒グラフで比較
# ======================================
metrics = ["soft_accuracy", "strict_accuracy", "consistency"]
x = np.arange(len(results_df))  # モデル数
width = 0.25  # 棒の幅

fig, ax = plt.subplots(figsize=(8, 8))

ax.bar(x, results_df["soft_accuracy"], width=width, label="Soft Accuracy", alpha=0.7)
ax.bar(
    x - width,
    results_df["strict_accuracy"],
    width=width,
    label="Strict Accuracy",
    alpha=0.7,
)
ax.bar(
    x + width + 0.05,
    results_df["consistency"],
    width=width,
    label="Consistency",
    alpha=0.7,
)

ax.set_xticks(x)
ax.set_xticklabels(results_df["model"], rotation=90)
ax.set_ylim(0, 1)
ax.set_ylabel("Rate")
ax.set_title("Soft/Strict Accuracy & Consistency by Model")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, "accuracy_consistency_comparison.png"))
plt.close()
