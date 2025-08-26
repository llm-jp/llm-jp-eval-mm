import os
import json
import argparse
from dataclasses import asdict
from loguru import logger

import eval_mm
import eval_mm.metrics
from utils import GenerationConfig
from base_vllm import VLLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
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
    
    parser.add_argument(
        "--gpu_memory_utilization", 
        type=float, 
        default=0.95,
        help="GPU memory utilization for vLLM (default: 0.95)"
    )
    parser.add_argument(
        "--max_model_len", 
        type=int, 
        default=None,
        help="Maximum model context length. If not specified, will use model's default"
    )
    parser.add_argument(
        "--tensor_parallel_size", 
        type=int, 
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1)"
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
    logger.info(f"Using GPU memory utilization: {args.gpu_memory_utilization}")
    logger.info(f"Using tensor parallel size: {args.tensor_parallel_size}")
    if args.max_model_len:
        logger.info(f"Using max model length: {args.max_model_len}")
    
    model = VLLM(
        model_id=args.model_id,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    preds = []

    qids = [task.doc_to_id(doc) for doc in task.dataset]
    images = [task.doc_to_visual(doc) for doc in task.dataset]
    texts = [task.doc_to_text(doc).replace("<image>", "") for doc in task.dataset]

    preds = model.batch_generate(images, texts, gen_kwargs)
    preds = [{"question_id": qid, "text": pred} for qid, pred in zip(qids, preds)]

    save_jsonl(prediction_path, preds)
    logger.info(f"Predictions saved to {prediction_path}")
    return preds, []


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

    output_dir = os.path.join(args.result_dir, args.task_id, args.model_id + "_vllm")
    os.makedirs(output_dir, exist_ok=True)

    preds, _ = load_or_generate_predictions(args, task, gen_kwargs, output_dir)

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