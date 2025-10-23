import argparse

from examples.model_table import get_class_from_model_id, get_model_spec

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
parser.add_argument(
    "--runtime",
    choices=["transformers", "vllm", "api"],
    help="実行したいランタイム。省略時はモデルのデフォルトを使用します。",
)
parser.add_argument(
    "--gpu_memory_utilization",
    type=float,
    default=0.8,
    help="examples.runtimes.vllm.base.VLLM を使う場合の GPU メモリ使用率",
)
parser.add_argument(
    "--tensor_parallel_size",
    type=int,
    default=1,
    help="examples.runtimes.vllm.base.VLLM を使う場合のテンソル並列数",
)

args = parser.parse_args()

spec = get_model_spec(args.model_id)
runtime = args.runtime or spec.default_runtime
if not spec.has_runtime(runtime):
    available = ", ".join(sorted(spec.runtimes))
    msg = f"{args.model_id} はランタイム '{runtime}' をサポートしていません (available: {available})"
    raise SystemExit(msg)

runtime_config = spec.get_runtime_config(runtime)
model_cls = get_class_from_model_id(args.model_id, runtime=runtime)

if (
    runtime == "vllm"
    and runtime_config.module_path == "examples.runtimes.vllm.base.VLLM"
):
    model = model_cls(
        model_id=args.model_id,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
else:
    model = model_cls(args.model_id)

model.test_vlm()
