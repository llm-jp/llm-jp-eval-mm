import argparse
from model_table import get_class_from_model_id

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
parser.add_argument(
    "--smoke-test-mode",
    choices=("offline", "online"),
    default="offline",
    help="Use local fixtures by default. Select online to fetch remote images.",
)

args = parser.parse_args()

model = get_class_from_model_id(args.model_id)(args.model_id)
model.test_vlm(smoke_test_mode=args.smoke_test_mode)
