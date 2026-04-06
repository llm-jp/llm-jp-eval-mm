"""Interactive prediction browser (Streamlit).

Usage:
    uv run streamlit run scripts/browse_prediction.py -- \
        --task_id japanese-heron-bench \
        --result_dir result \
        --model_list model-a model-b

Displays per-sample predictions with:
- Multi-image support (shows all images, not just the first)
- Side-by-side model comparison
- Score display per metric
- Paging and random jump navigation
"""

import os
import random
from argparse import ArgumentParser

import streamlit as st

import eval_mm
from eval_mm import load_predictions
from eval_mm.metadata import TASKS


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


def format_scores(pred: dict, model_id: str) -> str:
    """Extract score fields from a prediction record for display."""
    score_parts = []
    for key, val in pred.items():
        if key in ("question_id", "text", "answer", "input_text"):
            continue
        if isinstance(val, (int, float)):
            score_parts.append(f"**{key}**: {val:.2f}")
        elif isinstance(val, dict) and "score" in val:
            score_parts.append(f"**{key}**: {val['score']}")
    return " | ".join(score_parts) if score_parts else ""


if __name__ == "__main__":
    args = parse_args()

    task = eval_mm.TaskRegistry().load_task(args.task_id)
    task_meta = TASKS.get(args.task_id)
    task_display = task_meta.display_name if task_meta else args.task_id

    # Load model predictions
    predictions_per_model = {}
    for model_id in args.model_list:
        output_dir = os.path.join(args.result_dir, args.task_id, model_id)
        if not os.path.isfile(os.path.join(output_dir, "prediction.jsonl")):
            st.warning(f"Predictions not found for {model_id}")
            continue
        predictions_per_model[model_id] = load_predictions(output_dir)

    active_models = list(predictions_per_model.keys())

    ds = task.dataset
    st.set_page_config(layout="wide")
    if "page" not in st.session_state:
        st.session_state.page = 0

    SAMPLES_PER_PAGE = 30
    column_width_list = [1, 3, 3, 3] + [4] * len(active_models)
    st.write(f"# {task_display} ({args.task_id})")
    st.caption(f"{len(ds)} samples | {len(active_models)} models loaded")

    def show_sample(idx):
        sample = ds[idx]
        cols = st.columns(column_width_list)
        cols[0].markdown(task.doc_to_id(sample))

        # Multi-image support: show all images, not just the first
        images = task.doc_to_visual(sample)
        if images:
            for img in images:
                cols[1].image(img, width=250)
        else:
            cols[1].markdown("*(no image)*")

        cols[2].markdown(
            scrollable_text(task.doc_to_text(sample)), unsafe_allow_html=True
        )
        cols[3].markdown(
            scrollable_text(task.doc_to_answer(sample)), unsafe_allow_html=True
        )
        for i, model_id in enumerate(active_models):
            pred = predictions_per_model[model_id][idx]
            pred_text = pred.get("text", "")
            scores_str = format_scores(pred, model_id)
            display = pred_text
            if scores_str:
                display = f"{pred_text}\n\n---\n{scores_str}"
            cols[4 + i].markdown(
                scrollable_text(display), unsafe_allow_html=True
            )

    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    if nav_col1.button(f"Prev {SAMPLES_PER_PAGE}"):
        st.session_state.page = max(st.session_state.page - 1, 0)
    if nav_col2.button("Random"):
        st.session_state.page = random.randint(0, len(ds) // SAMPLES_PER_PAGE)
    if nav_col3.button(f"Next {SAMPLES_PER_PAGE}"):
        st.session_state.page = min(
            st.session_state.page + 1, len(ds) // SAMPLES_PER_PAGE
        )

    start_idx = st.session_state.page * SAMPLES_PER_PAGE
    end_idx = min(start_idx + SAMPLES_PER_PAGE, len(ds))

    st.write(f"### Showing samples {start_idx + 1} to {end_idx} / {len(ds)}")

    # Header
    header_cols = st.columns(column_width_list)
    header_cols[0].markdown("**ID**")
    header_cols[1].markdown("**Image(s)**")
    header_cols[2].markdown("**Question**")
    header_cols[3].markdown("**Answer**")
    for i, model_id in enumerate(active_models):
        header_cols[4 + i].markdown(f"**{model_id}**")

    for idx in range(start_idx, end_idx):
        with st.container():
            show_sample(idx)
            st.markdown("---")
