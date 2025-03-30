import os
import json
import argparse
from dataclasses import asdict
from tqdm import tqdm
from loguru import logger

import eval_mm
import eval_mm.metrics
from utils import GenerationConfig
from model_table import get_class_from_model_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument(
        "--task_id",
        default="japanese-heron-bench",
        help=f"Task ID to evaluate. Available: {eval_mm.TaskRegistry().get_task_list()}",
    )
    parser.add_argument("--judge_model", default="gpt-4o-2024-11-20")
    parser.add_argument("--batch_size_for_evaluation", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--max_dataset_len", type=int)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["heron-bench"],
        help=f"Metrics to evaluate. Available: {eval_mm.ScorerRegistry().get_metric_list()}",
    )
    parser.add_argument(
        "--rotate_choices", action="store_true", help="This option is used in MECHA-ja"
    )
    parser.add_argument(
        "--random_choice",
        action="store_true",
        help="If set, randomly choose the answer from the candidates when parse error occurs in JMMMU and MMMU tasks",
    )
    return parser.parse_args()


def load_or_generate_predictions(args, task, gen_kwargs, output_dir):
    prediction_path = os.path.join(output_dir, "prediction.jsonl")
    if os.path.exists(prediction_path) and not args.overwrite:
        logger.info(f"Loading predictions from {prediction_path}")
        with open(prediction_path) as f:
            preds = [json.loads(line) for line in f]
        assert len(preds) == len(
            task.dataset
        ), "Prediction length mismatch with dataset"
        return preds, []

    logger.info("Generating predictions...")
    model = get_class_from_model_id(args.model_id)(args.model_id)
    preds, errors = [], []
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

        if error_count > len(task.dataset) * 0.1:
            logger.error("Error count exceeded 10%. Terminating.")
            save_jsonl(os.path.join(output_dir, "error_message.jsonl"), errors)
            exit()

    save_jsonl(prediction_path, preds)
    if errors:
        save_jsonl(os.path.join(output_dir, "error_message.jsonl"), errors)
    logger.info(f"Predictions saved to {prediction_path}")
    return preds, errors


def save_jsonl(path, data):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def evaluate(args, task, preds, metrics):
    logger.info("Starting evaluation...")
    scores_by_metric = {}
    aggregated_metrics = {}

    for metric in metrics:
        scorer = eval_mm.ScorerRegistry.load_scorer(
            metric,
            eval_mm.ScorerConfig(
                docs=task.dataset,
                judge_model=args.judge_model,
                batch_size=args.batch_size_for_evaluation,
                client=eval_mm.OpenAIChatAPI(),
                random_choice=args.random_choice,
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


def save_final_results(preds, task, metrics, scores_by_metric, output_path):
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

    save_jsonl(output_path, final_results)
    logger.info(f"Final prediction with scores saved to {output_path}")


def main():
    args = parse_args()

    gen_kwargs = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        use_cache=args.use_cache,
    )

    task_config = eval_mm.TaskConfig(
        max_dataset_len=args.max_dataset_len,
        rotate_choices=args.rotate_choices,
    )
    task = eval_mm.TaskRegistry.load_task(args.task_id, task_config)

    output_dir = os.path.join(args.result_dir, args.task_id, args.model_id)
    os.makedirs(output_dir, exist_ok=True)

    preds, errors = load_or_generate_predictions(args, task, gen_kwargs, output_dir)

    if args.inference_only:
        logger.info("Inference only mode. Skipping evaluation.")
        return

    scores_by_metric, aggregated_metrics = evaluate(args, task, preds, args.metrics)

    prediction_path = os.path.join(output_dir, "prediction.jsonl")
    save_final_results(preds, task, args.metrics, scores_by_metric, prediction_path)

    evaluation_path = os.path.join(output_dir, "evaluation.jsonl")
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(aggregated_metrics, ensure_ascii=False) + "\n")
    logger.info(f"Evaluation result saved to {evaluation_path}")


if __name__ == "__main__":
    main()
