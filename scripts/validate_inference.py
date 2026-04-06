#!/usr/bin/env python3
"""Validate model inference and dataset format across all tasks/models.

Usage:
    # Full validation (all models load + all tasks with one model)
    uv run python scripts/validate_inference.py --result_dir result_validation

    # Dataset-only validation (no GPU needed)
    uv run python scripts/validate_inference.py --dataset-only

    # Single model test
    uv run python scripts/validate_inference.py --model "Qwen/Qwen2.5-VL-3B-Instruct"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Ensure HF_HOME is not under $HOME before any HF imports
_home = os.path.expanduser("~")
_hf_home = os.environ.get("HF_HOME", "")
if not _hf_home or _hf_home.startswith(_home):
    print(f"ERROR: HF_HOME must be set to a path outside $HOME.")
    print(f"  Current HF_HOME: {_hf_home or '(not set)'}")
    print(f"  $HOME: {_home}")
    print(f"  Fix: export HF_HOME=/path/to/shared/cache/huggingface")
    sys.exit(1)
print(f"OK: HF_HOME={_hf_home}")

# Add examples/ to path for model adapters
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "examples"))

import eval_mm
from eval_mm.metadata import TASKS, LEADERBOARD_MODELS


def validate_dataset(task_id: str) -> dict:
    """Validate that a task's dataset loads and formats correctly."""
    result = {"task_id": task_id, "status": "unknown", "error": None, "samples": 0}
    try:
        task = eval_mm.TaskRegistry.load_task(task_id)
        dataset = task.dataset

        if len(dataset) == 0:
            result["status"] = "empty"
            result["error"] = "Dataset has 0 samples"
            return result

        result["samples"] = len(dataset)

        # Validate first sample
        sample = dataset[0]

        # doc_to_text
        text = task.doc_to_text(sample)
        assert isinstance(text, str), f"doc_to_text returned {type(text)}, expected str"
        assert len(text) > 0, "doc_to_text returned empty string"

        # doc_to_visual
        images = task.doc_to_visual(sample)
        assert isinstance(images, list), f"doc_to_visual returned {type(images)}, expected list"

        # doc_to_answer
        answer = task.doc_to_answer(sample)
        assert isinstance(answer, (str, list)), f"doc_to_answer returned {type(answer)}"

        result["status"] = "ok"
        result["text_len"] = len(text)
        result["num_images"] = len(images)
        result["answer_type"] = type(answer).__name__

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


def validate_model_load(model_id: str, tp_size: int = 1) -> dict:
    """Validate that a model can be loaded with vLLM and generate 1 output."""
    result = {"model_id": model_id, "status": "unknown", "error": None, "load_time": 0}

    # Skip API models
    if not "/" in model_id:
        result["status"] = "skip"
        result["error"] = "API model (not local)"
        return result

    try:
        from base_vllm import VLLM

        t0 = time.time()
        model = VLLM(
            model_id=model_id,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=tp_size,
        )
        result["load_time"] = round(time.time() - t0, 1)

        # Try generating 1 token to confirm model works
        from PIL import Image
        from eval_mm.models.generation_config import GenerationConfig
        dummy_img = Image.new("RGB", (64, 64), color="white")
        output = model.generate(
            images=[dummy_img], text="Describe this image.",
            gen_kwargs=GenerationConfig(max_new_tokens=10),
        )
        assert isinstance(output, str), f"generate returned {type(output)}"

        result["status"] = "ok"
        result["output_preview"] = output[:100]

        # Cleanup
        del model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


def validate_inference(model_id: str, task_id: str, max_samples: int = 2) -> dict:
    """Run inference on a few samples to validate the full pipeline."""
    result = {
        "model_id": model_id,
        "task_id": task_id,
        "status": "unknown",
        "error": None,
        "num_processed": 0,
    }

    try:
        from base_vllm import VLLM

        model = VLLM(
            model_id=model_id,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
        )

        task = eval_mm.TaskRegistry.load_task(task_id)
        dataset = task.dataset

        n = min(max_samples, len(dataset))
        for i in range(n):
            sample = dataset[i]
            text = task.doc_to_text(sample)
            images = task.doc_to_visual(sample)

            from eval_mm.models.generation_config import GenerationConfig
            output = model.generate(
                images=images if images else None,
                text=text,
                gen_kwargs=GenerationConfig(max_new_tokens=64),
            )
            assert isinstance(output, str)
            result["num_processed"] += 1

        result["status"] = "ok"

        del model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate inference pipeline")
    parser.add_argument("--dataset-only", action="store_true",
                        help="Only validate dataset format (no GPU needed)")
    parser.add_argument("--model", type=str, default=None,
                        help="Test a specific model only")
    parser.add_argument("--task", type=str, default=None,
                        help="Test a specific task only")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--result_dir", type=str, default="result_validation",
                        help="Directory to save validation results")
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    report = {"hf_home": _hf_home, "datasets": [], "models": [], "inference": []}

    # Phase 1: Dataset validation
    print("\n" + "=" * 60)
    print("Phase 1: Dataset Format Validation")
    print("=" * 60)

    task_ids = [args.task] if args.task else list(TASKS.keys())
    for task_id in task_ids:
        print(f"\n  [{task_id}] ", end="", flush=True)
        r = validate_dataset(task_id)
        print(f"{'OK' if r['status'] == 'ok' else 'FAIL'} ({r['samples']} samples)")
        if r["error"]:
            print(f"    ERROR: {r['error']}")
        report["datasets"].append(r)

    ok = sum(1 for r in report["datasets"] if r["status"] == "ok")
    total = len(report["datasets"])
    print(f"\n  Dataset results: {ok}/{total} passed")

    if args.dataset_only:
        _save_report(report, args.result_dir)
        return

    # Phase 2: Model load validation
    print("\n" + "=" * 60)
    print("Phase 2: Model Load Validation (vLLM)")
    print("=" * 60)

    models = [args.model] if args.model else LEADERBOARD_MODELS
    for model_id in models:
        print(f"\n  [{model_id}] ", end="", flush=True)
        r = validate_model_load(model_id, tp_size=args.tp)
        status_str = r["status"].upper()
        if r["status"] == "ok":
            status_str += f" ({r['load_time']}s)"
        print(status_str)
        if r["error"]:
            print(f"    ERROR: {r['error']}")
        report["models"].append(r)

    ok = sum(1 for r in report["models"] if r["status"] == "ok")
    skip = sum(1 for r in report["models"] if r["status"] == "skip")
    fail = sum(1 for r in report["models"] if r["status"] == "error")
    print(f"\n  Model results: {ok} ok, {skip} skipped, {fail} failed")

    # Phase 3: Full pipeline validation (1 model x all tasks)
    print("\n" + "=" * 60)
    print("Phase 3: Full Pipeline Validation")
    print("=" * 60)

    test_model = args.model or "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"  Using model: {test_model}")

    for task_id in task_ids:
        print(f"\n  [{test_model} x {task_id}] ", end="", flush=True)
        r = validate_inference(test_model, task_id, max_samples=2)
        print(f"{'OK' if r['status'] == 'ok' else 'FAIL'} ({r['num_processed']} samples)")
        if r["error"]:
            print(f"    ERROR: {r['error']}")
        report["inference"].append(r)

    ok = sum(1 for r in report["inference"] if r["status"] == "ok")
    print(f"\n  Inference results: {ok}/{len(report['inference'])} passed")

    _save_report(report, args.result_dir)


def _save_report(report: dict, result_dir: str):
    report_path = os.path.join(result_dir, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    ds_ok = sum(1 for r in report["datasets"] if r["status"] == "ok")
    ds_total = len(report["datasets"])
    print(f"  Datasets:  {ds_ok}/{ds_total}")

    if report["models"]:
        m_ok = sum(1 for r in report["models"] if r["status"] == "ok")
        m_skip = sum(1 for r in report["models"] if r["status"] == "skip")
        m_fail = sum(1 for r in report["models"] if r["status"] == "error")
        print(f"  Models:    {m_ok} ok, {m_skip} skipped, {m_fail} failed")

    if report["inference"]:
        i_ok = sum(1 for r in report["inference"] if r["status"] == "ok")
        i_total = len(report["inference"])
        print(f"  Inference: {i_ok}/{i_total}")


if __name__ == "__main__":
    main()
