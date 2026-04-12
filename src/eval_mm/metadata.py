"""Single source of truth for task, metric, and model metadata.

Every display name, cluster assignment, dataset URL, and default metric
is defined here. CLI, leaderboard scripts, and web frontends should all
read from this module instead of maintaining their own copies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskMeta:
    task_id: str
    display_name: str
    cluster: str
    default_metrics: list[str] = field(default_factory=list)
    dataset_url: str = ""


@dataclass(frozen=True)
class MetricMeta:
    metric_id: str
    display_name: str


# ── Task metadata ──────────────────────────────────────────────────

TASKS: dict[str, TaskMeta] = {t.task_id: t for t in [
    # English (leaderboard display order)
    TaskMeta("okvqa", "OK-VQA", "英語",
             ["substring-match"], ""),
    TaskMeta("textvqa", "TextVQA", "英語",
             ["substring-match"], "https://huggingface.co/datasets/lmms-lab/textvqa"),
    TaskMeta("ai2d", "AI2D", "英語",
             ["ai2d"], "https://huggingface.co/datasets/lmms-lab/ai2d"),
    TaskMeta("chartqa", "ChartQA", "英語",
             ["substring-match"], "https://huggingface.co/datasets/lmms-lab/ChartQA"),
    TaskMeta("docvqa", "DocVQA", "英語",
             ["substring-match"], "https://huggingface.co/datasets/lmms-lab/DocVQA"),
    TaskMeta("blink", "BLINK", "英語",
             ["blink"], "https://huggingface.co/datasets/BLINK-Benchmark/BLINK"),
    TaskMeta("infographicvqa", "InfoVQA", "英語",
             ["substring-match"], "https://huggingface.co/datasets/lmms-lab/infographicvqa"),
    TaskMeta("mmmu", "MMMU", "英語",
             ["mmmu"], "https://huggingface.co/datasets/MMMU/MMMU"),
    TaskMeta("llava-bench-in-the-wild", "LLAVA", "英語",
             ["llm-as-a-judge", "rougel"], "https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild"),
    TaskMeta("chartqapro", "ChartQAPro", "英語",
             ["substring-match"], ""),
    TaskMeta("mmmlu", "MMMLU", "英語",
             ["exact-match"], ""),
    # Japanese (leaderboard display order)
    TaskMeta("cvqa", "CVQA", "視覚中心",
             ["substring-match"], "https://huggingface.co/datasets/afaji/cvqa"),
    TaskMeta("cc-ocr", "CC-OCR", "言語・知識中心",
             ["cc-ocr"], "https://huggingface.co/datasets/wulipc/CC-OCR"),
    TaskMeta("jic-vqa", "JIC", "視覚中心",
             ["jic-vqa"], "https://huggingface.co/datasets/line-corporation/JIC-VQA"),
    TaskMeta("ja-multi-image-vqa", "MulIm-VQA", "その他",
             ["llm-as-a-judge", "rougel"], "https://huggingface.co/datasets/SakanaAI/JA-Multi-Image-VQA"),
    TaskMeta("jmmmu", "JMMMU", "言語・知識中心",
             ["jmmmu"], "https://huggingface.co/datasets/JMMMU/JMMMU"),
    TaskMeta("jdocqa", "JDocQA", "言語・知識中心",
             ["jdocqa"], "https://github.com/mizuumi/JDocQA"),
    TaskMeta("ja-vlm-bench-in-the-wild", "JVB-ItW", "視覚中心",
             ["llm-as-a-judge", "rougel"], "https://huggingface.co/datasets/SakanaAI/JA-VLM-Bench-In-the-Wild"),
    TaskMeta("ja-vg-vqa-500", "VG-VQA", "視覚中心",
             ["llm-as-a-judge", "rougel"], "https://huggingface.co/datasets/SakanaAI/JA-VG-VQA-500"),
    TaskMeta("japanese-heron-bench", "Heron", "視覚中心",
             ["heron-bench"], "https://huggingface.co/datasets/turing-motors/Japanese-Heron-Bench"),
    TaskMeta("mecha-ja", "MECHA", "言語・知識中心",
             ["mecha-ja"], "https://huggingface.co/datasets/llm-jp/MECHA-ja"),
    TaskMeta("jawildtext-board-vqa", "JWTxt-Board", "言語・知識中心",
             ["jawildtext-board-vqa"], "https://huggingface.co/datasets/llm-jp/jawildtext"),
    TaskMeta("jawildtext-handwriting-ocr", "JWTxt-HW", "言語・知識中心",
             ["jawildtext-handwriting-ocr"], "https://huggingface.co/datasets/llm-jp/jawildtext"),
    TaskMeta("jawildtext-receipt-kie", "JWTxt-Receipt", "言語・知識中心",
             ["jawildtext-receipt-kie"], "https://huggingface.co/datasets/llm-jp/jawildtext"),
]}


# ── Metric metadata ───────────────────────────────────────────────

METRICS: dict[str, MetricMeta] = {m.metric_id: m for m in [
    MetricMeta("heron-bench", "LLM"),
    MetricMeta("llm-as-a-judge", "LLM"),
    MetricMeta("rougel", "Rouge"),
    MetricMeta("exact-match", "EM"),
    MetricMeta("substring-match", "Acc"),
    MetricMeta("jdocqa", "Acc"),
    MetricMeta("jmmmu", "Acc"),
    MetricMeta("jic-vqa", "Acc"),
    MetricMeta("mecha-ja", "Acc"),
    MetricMeta("mmmu", "Acc"),
    MetricMeta("cc-ocr", "macro_f1"),
    MetricMeta("ai2d", "Acc"),
    MetricMeta("blink", "Acc"),
    MetricMeta("mathvista", "Acc"),
    MetricMeta("jawildtext-board-vqa", "LLM"),
    MetricMeta("jawildtext-handwriting-ocr", "CER"),
    MetricMeta("jawildtext-receipt-kie", "F1"),
]}


# ── Leaderboard model list ────────────────────────────────────────

LEADERBOARD_MODELS: list[str] = [
    "stabilityai/japanese-stable-vlm",
    "cyberagent/llava-calm2-siglip",
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "neulab/Pangea-7B-hf",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "OpenGVLab/InternVL3-1B",
    "OpenGVLab/InternVL3-2B",
    "OpenGVLab/InternVL3-8B",
    "OpenGVLab/InternVL3-9B",
    "OpenGVLab/InternVL3-14B",
    "OpenGVLab/InternVL3-38B",
    "OpenGVLab/InternVL3-78B",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "gpt-4o-2024-11-20",
    "llm-jp/llm-jp-3-vila-14b",
    "SakanaAI/Llama-3-EvoVLM-JP-v2",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "google/gemma-3-4b-pt",
    "google/gemma-3-12b-pt",
    "google/gemma-3-27b-pt",
    "sbintuitions/sarashina2-vision-8b",
    "sbintuitions/sarashina2-vision-14b",
    "microsoft/Phi-4-multimodal-instruct",
    "MIL-UT/Asagi-14B",
    "turing-motors/Heron-NVILA-Lite-1B",
    "turing-motors/Heron-NVILA-Lite-2B",
    "turing-motors/Heron-NVILA-Lite-15B",
    "turing-motors/Heron-NVILA-Lite-33B",
    "CohereLabs/aya-vision-8b",
    "CohereLabs/aya-vision-32b",
]


# ── Model display names (for TeX / leaderboard rendering) ────────

MODEL_DISPLAY_NAMES: dict[str, str] = {
    # Domestic (Japanese) models
    "cyberagent/llava-calm2-siglip": "Llava-calm2-siglip",
    "SakanaAI/Llama-3-EvoVLM-JP-v2": "Llama-3-EvoVLM-JP-v2",
    "MIL-UT/Asagi-14B": "Asagi-14B",
    "sbintuitions/sarashina2-vision-8b": "Sarashina2-Vision-8b",
    "sbintuitions/sarashina2-vision-14b": "Sarashina2-Vision-14b",
    "turing-motors/Heron-NVILA-Lite-1B": "Heron-NVILA-Lite-1B",
    "turing-motors/Heron-NVILA-Lite-2B": "Heron-NVILA-Lite-2B",
    "turing-motors/Heron-NVILA-Lite-15B": "Heron-NVILA-Lite-15B",
    "turing-motors/Heron-NVILA-Lite-33B": "Heron-NVILA-Lite-33B",
    "llm-jp/llm-jp-3-vila-14b": "llm-jp-3-vila-14b",
    # Foreign models
    "llava-hf/llava-1.5-7b-hf": "Llava-1.5-7b",
    "llava-hf/llava-v1.6-mistral-7b-hf": "Llava-v1.6-mistral-7b",
    "CohereLabs/aya-vision-8b": "Aya-Vision-8b",
    "CohereLabs/aya-vision-32b": "Aya-Vision-32b",
    "neulab/Pangea-7B-hf": "Pangea-7B",
    "microsoft/Phi-4-multimodal-instruct": "Phi-4-multimodal-instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "Llama-3.2-90B-Vision-Instruct",
    "google/gemma-3-4b-it": "Gemma-3-4b-it",
    "google/gemma-3-12b-it": "Gemma-3-12b-it",
    "google/gemma-3-27b-it": "Gemma-3-27b-it",
    "OpenGVLab/InternVL3-1B": "InternVL3-1B",
    "OpenGVLab/InternVL3-2B": "InternVL3-2B",
    "OpenGVLab/InternVL3-8B": "InternVL3-8B",
    "OpenGVLab/InternVL3-14B": "InternVL3-14B",
    "OpenGVLab/InternVL3-38B": "InternVL3-38B",
    "OpenGVLab/InternVL3-78B": "InternVL3-78B",
    "Qwen/Qwen2-VL-7B-Instruct": "Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct": "Qwen2-VL-72B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B-Instruct",
}

# Domestic model IDs in display order for TeX tables.
DOMESTIC_MODEL_ORDER: list[str] = [
    "cyberagent/llava-calm2-siglip",
    "SakanaAI/Llama-3-EvoVLM-JP-v2",
    "MIL-UT/Asagi-14B",
    "sbintuitions/sarashina2-vision-8b",
    "sbintuitions/sarashina2-vision-14b",
    "turing-motors/Heron-NVILA-Lite-1B",
    "turing-motors/Heron-NVILA-Lite-2B",
    "turing-motors/Heron-NVILA-Lite-15B",
    "turing-motors/Heron-NVILA-Lite-33B",
    "llm-jp/llm-jp-3-vila-14b",
]

# Foreign model IDs in display order for TeX tables.
FOREIGN_MODEL_ORDER: list[str] = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "CohereLabs/aya-vision-8b",
    "CohereLabs/aya-vision-32b",
    "neulab/Pangea-7B-hf",
    "microsoft/Phi-4-multimodal-instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "OpenGVLab/InternVL3-1B",
    "OpenGVLab/InternVL3-2B",
    "OpenGVLab/InternVL3-8B",
    "OpenGVLab/InternVL3-14B",
    "OpenGVLab/InternVL3-38B",
    "OpenGVLab/InternVL3-78B",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
]


# ── Convenience accessors ─────────────────────────────────────────

def get_task_alias() -> dict[str, str]:
    """Return {task_id: display_name} mapping."""
    return {t.task_id: t.display_name for t in TASKS.values()}


def get_task_cluster_alias() -> dict[str, str]:
    """Return {display_name: cluster} mapping."""
    return {t.display_name: t.cluster for t in TASKS.values()}


def get_tasks_by_language() -> tuple[list[str], list[str]]:
    """Return (english_display_names, japanese_display_names) derived from cluster field."""
    english = [t.display_name for t in TASKS.values() if t.cluster == "英語"]
    japanese = [t.display_name for t in TASKS.values() if t.cluster != "英語"]
    return english, japanese


def get_metric_alias() -> dict[str, str]:
    """Return {metric_id: display_name} mapping."""
    return {m.metric_id: m.display_name for m in METRICS.values()}


def generate_default_metrics_json() -> dict:
    """Generate the default_metrics.json content for github_pages."""
    result: dict[str, str] = {}
    for task in TASKS.values():
        if task.default_metrics:
            metric_id = task.default_metrics[0]
            metric_display = METRICS[metric_id].display_name if metric_id in METRICS else metric_id
            # Use a simplified display: "overall" for heron-bench, "rouge" for rougel, else display_name
            if metric_id == "heron-bench":
                result[task.display_name] = "overall"
            elif metric_id == "rougel":
                result[task.display_name] = "rouge"
            elif metric_id == "llm-as-a-judge":
                result[task.display_name] = "rouge"
            else:
                result[task.display_name] = metric_display
    return {"default_metrics": result}


def generate_dataset_url_json() -> dict:
    """Generate the dataset_url.json content for github_pages."""
    result: dict[str, dict[str, str]] = {}
    for task in TASKS.values():
        if task.dataset_url:
            result[task.display_name] = {"url": task.dataset_url}
    return result


def write_github_pages_json(output_dir: str) -> None:
    """Write default_metrics.json and dataset_url.json to the given directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "default_metrics.json"), "w") as f:
        json.dump(generate_default_metrics_json(), f, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, "dataset_url.json"), "w") as f:
        json.dump(generate_dataset_url_json(), f, indent=4, ensure_ascii=False)
