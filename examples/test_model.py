import argparse
from model_table import get_class_from_model_id

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")

args = parser.parse_args()

model = get_class_from_model_id(args.model_id)(args.model_id)
model.test_vlm()
