import json
import os

import eval_mm
from tqdm import tqdm
import importlib
import argparse
import time
from utils import GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--class_path", type=str, default="llava_1_5_7b_hf")
parser.add_argument("--task_id", type=str, default="japanese-heron-bench")
parser.add_argument("--openai_model_id", type=str, default="gpt-4o-mini-2024-07-18")
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

args = parser.parse_args()

gen_kwargs = GenerationConfig(
    max_new_tokens=args.max_new_tokens,
    temperature=args.temperature,
    top_p=args.top_p,
    num_beams=args.num_beams,
    do_sample=args.do_sample,
    use_cache=args.use_cache,
)

class_path = args.class_path
task_id = args.task_id
openai_model_id = args.openai_model_id

module = importlib.import_module(class_path)
model_id = module.VLM.model_id.replace("/", "-")

task = eval_mm.api.registry.get_task(task_id)
dataset = task.dataset

# save the predictions to jsonl file
os.makedirs(args.result_dir, exist_ok=True)
result_dir = f"{args.result_dir}/{task_id}"
os.makedirs(result_dir, exist_ok=True)
prediction_result_dir = os.path.join(result_dir, "prediction")
os.makedirs(prediction_result_dir, exist_ok=True)
evaluation_result_dir = os.path.join(result_dir, "evaluation")
os.makedirs(evaluation_result_dir, exist_ok=True)

unix_time = int(time.time())

prediction_result_file_path = os.path.join(prediction_result_dir, f"{model_id}.jsonl")


# if prediciton is already done, load the prediction
if os.path.exists(prediction_result_file_path) and not args.overwrite:
    with open(prediction_result_file_path, "r") as f:
        preds = [json.loads(line) for line in f]
    print(f"Prediction result loaded from {prediction_result_file_path}")
else:
    model = module.VLM()
    preds = []
    for doc in tqdm(dataset):
        # print("doc", doc)
        image = task.doc_to_visual(doc)
        text = task.doc_to_text(doc)
        qid = task.doc_to_id(doc)
        # print("image", image)
        # print("text", text)
        # print("qid", qid)

        pred = {
            "question_id": qid,
            "text": model.generate(image, text, gen_kwargs),
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
metrics, eval_results = task.compute_metrics(
    preds, model_id=openai_model_id, batch_size=args.batch_size_for_evaluation
)


results = task.format_result(preds, eval_results)
with open(os.path.join(prediction_result_file_path), "w") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
print(f"Prediction result saved to {prediction_result_file_path}")

eval_result_file_path = os.path.join(evaluation_result_dir, f"{model_id}.jsonl")
with open(eval_result_file_path, "w") as f:
    f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

print(f"Metrics: {metrics}")
print(f"Evaluation result example: {eval_results[0]}")
