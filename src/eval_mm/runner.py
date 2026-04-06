"""Evaluation runner — the shared engine behind CLI and programmatic use."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import TYPE_CHECKING

from loguru import logger
from tqdm import tqdm

from .metrics.scorer import ScorerConfig
from .metrics.scorer_registry import ScorerRegistry
from .tasks.task import TaskConfig
from .tasks.task_registry import TaskRegistry
from .result_schema import write_manifest
from .utils.azure_client import OpenAIChatAPI

if TYPE_CHECKING:
    from .models.base_vlm import BaseVLM
    from .models.generation_config import GenerationConfig


def save_jsonl(path: str, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_predictions(prediction_path: str) -> list[dict]:
    with open(prediction_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_predictions_sequential(
    model: BaseVLM,
    task,
    gen_kwargs: GenerationConfig,
    output_dir: str,
    error_threshold: float = 0.1,
) -> tuple[list[dict], list[dict]]:
    """Generate predictions one-by-one (transformers backend)."""
    preds: list[dict] = []
    errors: list[dict] = []
    error_count = 0

    for doc in tqdm(task.dataset):
        qid = task.doc_to_id(doc)
        images = task.doc_to_visual(doc)
        text = task.doc_to_text(doc).replace("<image>", "")

        try:
            generated_text = model.generate(images, text, gen_kwargs)
        except Exception as e:
            logger.error(f"Error on {qid}: {e}")
            generated_text, error_count = "", error_count + 1
            errors.append({"question_id": qid, "error": str(e)})

        preds.append({"question_id": qid, "text": generated_text})

        if error_count > len(task.dataset) * error_threshold:
            logger.error("Error count exceeded threshold. Terminating.")
            save_jsonl(os.path.join(output_dir, "error_message.jsonl"), errors)
            raise RuntimeError(
                f"Error rate exceeded {error_threshold*100:.0f}%: "
                f"{error_count}/{len(task.dataset)}"
            )

    return preds, errors


def generate_predictions_batch(
    model: BaseVLM,
    task,
    gen_kwargs: GenerationConfig,
) -> list[dict]:
    """Generate predictions in batch (vLLM backend)."""
    qids = [task.doc_to_id(doc) for doc in task.dataset]
    images = [task.doc_to_visual(doc) for doc in task.dataset]
    texts = [task.doc_to_text(doc).replace("<image>", "") for doc in task.dataset]

    results = model.batch_generate(images, texts, gen_kwargs)
    return [{"question_id": qid, "text": pred} for qid, pred in zip(qids, results)]


def evaluate_predictions(
    task,
    preds: list[dict],
    metrics: list[str],
    judge_model: str = "gpt-4o-2024-11-20",
    batch_size: int = 10,
    random_choice: bool = False,
) -> tuple[dict[str, list], dict[str, dict]]:
    """Run scoring on predictions and return per-sample scores + aggregates."""
    logger.info("Starting evaluation...")
    scores_by_metric: dict[str, list] = {}
    aggregated_metrics: dict[str, dict] = {}

    for metric in metrics:
        scorer = ScorerRegistry.load_scorer(
            metric,
            ScorerConfig(
                docs=task.dataset,
                judge_model=judge_model,
                batch_size=batch_size,
                client=OpenAIChatAPI(),
                random_choice=random_choice,
            ),
        )
        scores = scorer.score(
            [task.doc_to_answer(doc) for doc in task.dataset],
            [pred["text"] for pred in preds],
        )
        scores_by_metric[metric] = scores
        aggregate = scorer.aggregate(scores)
        aggregated_metrics[metric] = asdict(aggregate)

        logger.info(f"Scores for {metric}: {scores}")
        logger.info(f"Aggregate for {metric}: {aggregate}")

    return scores_by_metric, aggregated_metrics


def save_results(
    preds: list[dict],
    task,
    metrics: list[str],
    scores_by_metric: dict[str, list],
    aggregated_metrics: dict[str, dict],
    output_dir: str,
) -> None:
    """Write prediction.jsonl (with scores) and evaluation.jsonl."""
    final_results = []
    for i, pred in enumerate(preds):
        doc = task.dataset[i]
        result = {
            "question_id": pred["question_id"],
            "text": pred["text"],
            "answer": task.doc_to_answer(doc),
            "input_text": task.doc_to_text(doc),
        }
        for metric in metrics:
            result[metric] = scores_by_metric[metric][i]
        final_results.append(result)

    prediction_path = os.path.join(output_dir, "prediction.jsonl")
    save_jsonl(prediction_path, final_results)
    logger.info(f"Final prediction with scores saved to {prediction_path}")

    evaluation_path = os.path.join(output_dir, "evaluation.jsonl")
    with open(evaluation_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(aggregated_metrics, ensure_ascii=False) + "\n")
    logger.info(f"Evaluation result saved to {evaluation_path}")


def run_evaluation(
    *,
    model: BaseVLM | None = None,
    model_id: str,
    task_id: str,
    metrics: list[str],
    gen_config: GenerationConfig | None = None,
    task_config: TaskConfig | None = None,
    result_dir: str = "result",
    judge_model: str = "gpt-4o-2024-11-20",
    batch_size_for_evaluation: int = 10,
    overwrite: bool = False,
    inference_only: bool = False,
    random_choice: bool = False,
    batch_mode: bool = False,
) -> dict[str, dict] | None:
    """Run the full evaluation pipeline.

    Parameters
    ----------
    model : BaseVLM | None
        Model instance. If *None*, predictions must already exist in result_dir.
    model_id : str
        Model identifier (used for output directory naming).
    task_id : str
        Registered task name.
    metrics : list[str]
        List of scorer names to evaluate.
    gen_config : GenerationConfig | None
        Generation parameters. Required when *model* is provided.
    task_config : TaskConfig | None
        Task-level configuration.
    result_dir : str
        Root directory for results.
    judge_model : str
        Model ID for LLM-as-a-judge.
    batch_size_for_evaluation : int
        Batch size for scorer API calls.
    overwrite : bool
        Regenerate predictions even if they exist.
    inference_only : bool
        Skip evaluation after generation.
    random_choice : bool
        Randomly choose answer on parse error (JMMMU/MMMU).
    batch_mode : bool
        Use batch generation (vLLM) instead of sequential.

    Returns
    -------
    dict | None
        Aggregated metrics dict, or *None* if inference_only.
    """
    from .models.generation_config import GenerationConfig as _GC

    if task_config is None:
        task_config = TaskConfig()
    if gen_config is None:
        gen_config = _GC()

    task = TaskRegistry.load_task(task_id, task_config)

    output_dir = os.path.join(result_dir, task_id, model_id)
    os.makedirs(output_dir, exist_ok=True)

    prediction_path = os.path.join(output_dir, "prediction.jsonl")

    # --- Load or generate predictions ---
    errors: list[dict] = []
    if os.path.exists(prediction_path) and not overwrite:
        logger.info(f"Loading predictions from {prediction_path}")
        preds = load_predictions(prediction_path)
        assert len(preds) == len(task.dataset), "Prediction length mismatch with dataset"
    elif model is not None:
        logger.info("Generating predictions...")
        if batch_mode:
            preds = generate_predictions_batch(model, task, gen_config)
        else:
            preds, errors = generate_predictions_sequential(
                model, task, gen_config, output_dir
            )
        save_jsonl(prediction_path, preds)
        if errors:
            save_jsonl(os.path.join(output_dir, "error_message.jsonl"), errors)
        logger.info(f"Predictions saved to {prediction_path}")
    else:
        raise FileNotFoundError(
            f"No predictions found at {prediction_path} and no model provided."
        )

    if inference_only:
        logger.info("Inference only mode. Skipping evaluation.")
        return None

    # --- Evaluate ---
    scores_by_metric, aggregated_metrics = evaluate_predictions(
        task,
        preds,
        metrics,
        judge_model=judge_model,
        batch_size=batch_size_for_evaluation,
        random_choice=random_choice,
    )

    save_results(preds, task, metrics, scores_by_metric, aggregated_metrics, output_dir)

    # Write manifest for downstream consumers (leaderboard, prediction browser)
    write_manifest(output_dir, model_id, task_id, metrics)

    return aggregated_metrics
