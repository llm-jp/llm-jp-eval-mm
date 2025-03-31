import streamlit as st
import random
import eval_mm
from argparse import ArgumentParser
import os
import json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--task_id", type=str, default="japanese-heron-bench")
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--model_list", type=str, nargs="+", default=[])

    return parser.parse_args()


def scrollable_text(text):
    return (
        f'<div style="max-height: 300px; overflow-y: auto; height: auto;">{text}</div>'
    )


if __name__ == "__main__":
    args = parse_args()

    task = eval_mm.TaskRegistry().load_task(args.task_id)

    # Load model prediction
    predictions_per_model = {}
    for model_id in args.model_list:
        prediction_path = os.path.join(
            args.result_dir, args.task_id, model_id, "prediction.jsonl"
        )
        with open(prediction_path, "r") as f:
            predictions_per_model[model_id] = [json.loads(line) for line in f]

    # VQAデータ読み込み
    ds = task.dataset
    # session_stateの初期化
    st.set_page_config(layout="wide")
    if "page" not in st.session_state:
        st.session_state.page = 0  # 現在のページ番号

    SAMPLES_PER_PAGE = 30  # 1ページに表示する件数
    # Question ID, Image, Question, Answer, Prediction_model1, Prediction_model2,..
    column_width_list = [1, 3, 3, 3] + [4] * len(args.model_list)
    st.write(f"# {args.task_id}")

    def show_sample(idx):
        sample = ds[idx]
        cols = st.columns(column_width_list)
        cols[0].markdown(task.doc_to_id(sample))
        cols[1].image(task.doc_to_visual(sample)[0], width=300)
        cols[2].markdown(
            scrollable_text(task.doc_to_text(sample)), unsafe_allow_html=True
        )
        cols[3].markdown(
            scrollable_text(task.doc_to_answer(sample)), unsafe_allow_html=True
        )
        for model_id in args.model_list:
            cols[4 + args.model_list.index(model_id)].markdown(
                scrollable_text(predictions_per_model[model_id][idx]["text"]),
                unsafe_allow_html=True,
            )

    # ナビゲーションボタン
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    if nav_col1.button(f"Prev {SAMPLES_PER_PAGE}"):
        st.session_state.page = max(st.session_state.page - 1, 0)
    if nav_col2.button("Random"):
        st.session_state.page = random.randint(0, len(ds) // SAMPLES_PER_PAGE)
    if nav_col3.button(f"Next {SAMPLES_PER_PAGE}"):
        st.session_state.page = min(
            st.session_state.page + 1, len(ds) // SAMPLES_PER_PAGE
        )

    # 現在のページのサンプルを表示
    start_idx = st.session_state.page * SAMPLES_PER_PAGE
    end_idx = min(start_idx + SAMPLES_PER_PAGE, len(ds))

    st.write(f"### Showing samples {start_idx + 1} to {end_idx} / {len(ds)}")

    # ヘッダー columnを表示
    header_cols = st.columns(column_width_list)
    header_cols[0].markdown("ID")
    header_cols[1].markdown("Image")
    header_cols[2].markdown("Question")
    header_cols[3].markdown("Answer")
    for model_id in args.model_list:
        header_cols[4 + args.model_list.index(model_id)].markdown(
            f"Prediction ({model_id})"
        )

    # サンプルを表示
    for idx in range(start_idx, end_idx):
        with st.container():
            show_sample(idx)
            st.markdown("---")
