import json
import os

import eval_mm
from tqdm import tqdm
import argparse
from utils import GenerationConfig
from model_table import get_class_from_model_id

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
    help="metrics to evaluate. You can specify multiple metrics separated by comma (e.g. --metrics exact_match,llm_as_a_judge) You can use rougel,substring_match,jmmmu,jdocqa,llm_as_a_judge_heron_bench,exact_match",
)
parser.add_argument(
    "--rotate_choices",
    action="store_true",
    help="If set, rotate the choices of MCQ options for evaluation.",
)

valid_metrics = [
    "rougel",
    "substring_match",
    "jmmmu",
    "jdocqa",
    "llm_as_a_judge_heron_bench",
    "exact_match",
    "llm_as_a_judge",
    "mmmu",
    "jic_vqa",
    "mecha-ja",
]


def validate_metrics(metrics: list[str]):
    for metric in metrics:
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric: {metric}. Valid metrics are {valid_metrics}"
            )


args = parser.parse_args()

gen_kwargs = GenerationConfig(
    max_new_tokens=args.max_new_tokens,
    temperature=args.temperature,
    top_p=args.top_p,
    num_beams=args.num_beams,
    do_sample=args.do_sample,
    use_cache=args.use_cache,
)

task_id = args.task_id

task_config = eval_mm.api.task.TaskConfig(
    max_dataset_len=args.max_dataset_len,
    judge_model=args.judge_model,
    batch_size_for_evaluation=args.batch_size_for_evaluation,
    rotate_choices=args.rotate_choices,
)
task = eval_mm.api.registry.get_task_cls(task_id)(task_config)

# save the predictions to jsonl file
os.makedirs(args.result_dir, exist_ok=True)
result_dir = f"{args.result_dir}/{task_id}"
os.makedirs(result_dir, exist_ok=True)
prediction_result_dir = os.path.join(result_dir, "prediction")
os.makedirs(prediction_result_dir, exist_ok=True)
evaluation_result_dir = os.path.join(result_dir, "evaluation")
os.makedirs(evaluation_result_dir, exist_ok=True)

prediction_result_file_path = os.path.join(
    prediction_result_dir, f"{args.model_id.replace('/', '-')}.jsonl"
)

# if prediciton is already done, load the prediction
if os.path.exists(prediction_result_file_path) and not args.overwrite:
    with open(prediction_result_file_path, "r") as f:
        preds = [json.loads(line) for line in f]
    assert (
        len(preds) == len(task.dataset)
    ), f"Prediction result length is not equal to the dataset length. Prediction result length: {len(preds)}, Dataset length: {len(task.dataset)}"
    print(f"Prediction result loaded from {prediction_result_file_path}")
else:
    model = get_class_from_model_id(args.model_id)(args.model_id)
    preds = []
    print(task.dataset)
    for doc in tqdm(task.dataset):
        image = task.doc_to_visual(doc)
        text = task.doc_to_text(doc)
        qid = task.doc_to_id(doc)
        try:
            generated_text = model.generate(image, text, gen_kwargs)
        except Exception as e:
            print(f"Error occurred for question_id: {qid}")
            print(e)
            generated_text = ""
        pred = {
            "question_id": qid,
            "text": generated_text,
        }
        preds.append(pred)
    with open(prediction_result_file_path, "w") as f:
        for pred in preds:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


if args.inference_only:
    print("Inference only mode. Skip evaluation.")
    exit()
print("Evaluation start")
# evaluate the predictions

metrics = args.metrics.split(",")
validate_metrics(metrics)

scores_for_each_metric = {}

for metric in metrics:
    scores_for_each_metric[metric] = task.calc_scores(preds, metric)
    print(f"Scores for {metric}: {scores_for_each_metric[metric]}")

calculated_metrics = {}

for metric in metrics:
    calculated_metrics[metric] = task.gather_scores(
        scores_for_each_metric[metric], metric
    )
    print(f"{metric}: {calculated_metrics[metric]}")


with open(prediction_result_file_path, "w") as f:
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
print(f"Prediction result saved to {prediction_result_file_path}")

eval_result_file_path = os.path.join(
    evaluation_result_dir, f"{args.model_id.replace('/', '-')}.json"
)
with open(eval_result_file_path, "w") as f:
    json.dump(calculated_metrics, ensure_ascii=False, indent=4, fp=f)
print(f"Evaluation result saved to {eval_result_file_path}")
