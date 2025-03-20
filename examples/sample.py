import json
import os

import eval_mm
from tqdm import tqdm
import argparse
import eval_mm.metrics
from utils import GenerationConfig
from model_table import get_class_from_model_id
from dataclasses import asdict
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--task_id", type=str, default="japanese-heron-bench")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--batch_size_for_evaluation", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument(
        "--max_dataset_len",
        type=int,
        default=None,
        help="max data size for evaluation. If None, use all data. Else, use the first n data.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="llm_as_a_judge_heron_bench",
        help=f"metrics to evaluate. You can specify multiple metrics separated by comma (e.g. --metrics exact_match,llm_as_a_judge) You can use the following metrics: {list(eval_mm.metrics.ScorerRegistry._scorers.keys())}",
    )
    parser.add_argument(
        "--rotate_choices",
        action="store_true",
        help="If set, rotate the choices of MCQ options for evaluation.",
    )
    return parser.parse_args()


def validate_metrics(metrics: list[str]):
    for metric in metrics:
        if metric not in list(eval_mm.metrics.ScorerRegistry._scorers.keys()):
            raise ValueError(
                f"Invalid metric: {metric}. You can use the following metrics: {list(eval_mm.metrics.ScorerRegistry._scorers.keys())}"
            )


if __name__ == "__main__":
    args = parse_args()
    metrics = args.metrics.split(",")
    validate_metrics(metrics)

    gen_kwargs = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        use_cache=args.use_cache,
    )

    task_config = eval_mm.api.task.TaskConfig(
        max_dataset_len=args.max_dataset_len,
        judge_model=args.judge_model,
        batch_size_for_evaluation=args.batch_size_for_evaluation,
        rotate_choices=args.rotate_choices,
    )
    task = eval_mm.api.registry.get_task_cls(args.task_id)(task_config)

    output_dir = os.path.join(args.result_dir, args.task_id, args.model_id)
    os.makedirs(output_dir, exist_ok=True)

    # if prediciton is already done, load the prediction
    prediction_path = os.path.join(output_dir, "prediction.jsonl")
    if os.path.exists(prediction_path) and not args.overwrite:
        with open(prediction_path, "r") as f:
            preds = [json.loads(line) for line in f]
        assert (
            len(preds) == len(task.dataset)
        ), f"Prediction result length is not equal to the dataset length. Prediction result length: {len(preds)}, Dataset length: {len(task.dataset)}"
        logger.info(f"Prediction result loaded from {prediction_path}")
    else:
        model = get_class_from_model_id(args.model_id)(args.model_id)
        preds = []
        logger.info(task.dataset)
        for doc in tqdm(task.dataset):
            images = task.doc_to_visual(doc)
            text = task.doc_to_text(doc)
            if "<image>" in text:
                text = text.replace("<image>", "")
            qid = task.doc_to_id(doc)
            generated_text = model.generate(images, text, gen_kwargs)
            pred = {
                "question_id": qid,
                "text": generated_text,
            }
            preds.append(pred)
        with open(prediction_path, "w") as f:
            for pred in preds:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    if args.inference_only:
        logger.info("Inference only mode. Skip evaluation.")
        exit()
    logger.info("Evaluation start")
    # evaluate the predictions

    scores_for_each_metric = {}
    calculated_metrics = {}

    for metric in metrics:
        scorer = eval_mm.metrics.ScorerRegistry.get_scorer(metric)
        scores = scorer.score(
            [task.doc_to_answer(doc) for doc in task.dataset],
            [pred["text"] for pred in preds],
            docs=task.dataset,
            client=task.client,
            judge_model=task.config.judge_model,
            batch_size=task.config.batch_size_for_evaluation,
        )
        logger.info(scores)
        scores_for_each_metric[metric] = scores
        logger.info(f"Scores for {metric}: {scores_for_each_metric[metric]}")
        aggregate_output = scorer.aggregate(scores, docs=task.dataset)

        calculated_metrics[metric] = asdict(aggregate_output)
        logger.info(f"Aggregate score for {metric}: {aggregate_output}")

    with open(prediction_path, "w") as f:
        for i, pred in enumerate(preds):
            question_id = pred["question_id"]
            text = pred["text"]
            answer = task.doc_to_answer(task.dataset[i])
            input_text = task.doc_to_text(task.dataset[i])
            content = {
                "question_id": question_id,
                "text": text,
                "answer": answer,
                "input_text": input_text,
            }
            for metric in metrics:
                content[metric] = scores_for_each_metric[metric][i]
            f.write(json.dumps(content, ensure_ascii=False) + "\n")
    logger.info(f"Prediction result saved to {prediction_path}")

    evaluation_path = os.path.join(output_dir, "evaluation.jsonl")
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(calculated_metrics, ensure_ascii=False) + "\n")
    logger.info(f"Evaluation result saved to {evaluation_path}")
