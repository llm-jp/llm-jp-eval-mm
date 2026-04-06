/**
 * Mock leaderboard data for development.
 *
 * Structure mirrors the metadata.py definitions: tasks are grouped by
 * cluster (英語 = English, others = Japanese-centric), and each model
 * entry carries per-task scores keyed by display_name.
 */

// ── Task definitions (mirrors metadata.py) ──────────────────────

export interface TaskDef {
  taskId: string;
  displayName: string;
  cluster: string;
}

export const TASKS: TaskDef[] = [
  // English
  { taskId: "okvqa", displayName: "OK-VQA", cluster: "英語" },
  { taskId: "textvqa", displayName: "TextVQA", cluster: "英語" },
  { taskId: "ai2d", displayName: "AI2D", cluster: "英語" },
  { taskId: "chartqa", displayName: "ChartQA", cluster: "英語" },
  { taskId: "docvqa", displayName: "DocVQA", cluster: "英語" },
  { taskId: "blink", displayName: "BLINK", cluster: "英語" },
  { taskId: "infographicvqa", displayName: "InfoVQA", cluster: "英語" },
  { taskId: "mmmu", displayName: "MMMU", cluster: "英語" },
  { taskId: "llava-bench-in-the-wild", displayName: "LLAVA", cluster: "英語" },
  { taskId: "chartqapro", displayName: "ChartQAPro", cluster: "英語" },
  { taskId: "mmmlu", displayName: "MMMLU", cluster: "英語" },
  // Japanese-centric
  { taskId: "cvqa", displayName: "CVQA", cluster: "視覚中心" },
  { taskId: "cc-ocr", displayName: "CC-OCR", cluster: "言語・知識中心" },
  { taskId: "jic-vqa", displayName: "JIC", cluster: "視覚中心" },
  {
    taskId: "ja-multi-image-vqa",
    displayName: "MulIm-VQA",
    cluster: "その他",
  },
  { taskId: "jmmmu", displayName: "JMMMU", cluster: "言語・知識中心" },
  { taskId: "jdocqa", displayName: "JDocQA", cluster: "言語・知識中心" },
  {
    taskId: "ja-vlm-bench-in-the-wild",
    displayName: "JVB-ItW",
    cluster: "視覚中心",
  },
  { taskId: "ja-vg-vqa-500", displayName: "VG-VQA", cluster: "視覚中心" },
  {
    taskId: "japanese-heron-bench",
    displayName: "Heron",
    cluster: "視覚中心",
  },
  { taskId: "mecha-ja", displayName: "MECHA", cluster: "言語・知識中心" },
];

export const EN_TASKS = TASKS.filter((t) => t.cluster === "英語");
export const JA_TASKS = TASKS.filter((t) => t.cluster !== "英語");

// ── Model entry ─────────────────────────────────────────────────

export interface ModelEntry {
  modelId: string;
  displayName: string;
  /** Per-task scores keyed by TaskDef.displayName. null = not evaluated. */
  scores: Record<string, number | null>;
  /** Aggregate scores computed from per-task values. */
  overall: number;
  enAvg: number;
  jaAvg: number;
}

// Helper: compute average of non-null values
function avg(vals: (number | null)[]): number {
  const valid = vals.filter((v): v is number => v !== null);
  if (valid.length === 0) return 0;
  return Math.round((valid.reduce((a, b) => a + b, 0) / valid.length) * 10) / 10;
}

function makeEntry(
  modelId: string,
  displayName: string,
  scores: Record<string, number | null>,
): ModelEntry {
  const enScores = EN_TASKS.map((t) => scores[t.displayName] ?? null);
  const jaScores = JA_TASKS.map((t) => scores[t.displayName] ?? null);
  const allScores = [...enScores, ...jaScores];
  return {
    modelId,
    displayName,
    scores,
    overall: avg(allScores),
    enAvg: avg(enScores),
    jaAvg: avg(jaScores),
  };
}

// ── Mock data ───────────────────────────────────────────────────

export const LEADERBOARD_DATA: ModelEntry[] = [
  makeEntry("Qwen/Qwen2.5-VL-72B-Instruct", "Qwen2.5-VL-72B", {
    "OK-VQA": 68.3, "TextVQA": 84.1, "AI2D": 88.5, "ChartQA": 86.2,
    "DocVQA": 93.1, "BLINK": 55.7, "InfoVQA": 76.4, "MMMU": 64.3,
    "LLAVA": 82.6, "ChartQAPro": 72.1, "MMMLU": 78.5,
    "CVQA": 61.2, "CC-OCR": 58.3, "JIC": 72.4, "MulIm-VQA": 65.8,
    "JMMMU": 52.1, "JDocQA": 71.3, "JVB-ItW": 78.9, "VG-VQA": 74.2,
    "Heron": 69.5, "MECHA": 54.7,
  }),
  makeEntry("OpenGVLab/InternVL3-78B", "InternVL3-78B", {
    "OK-VQA": 66.1, "TextVQA": 82.3, "AI2D": 86.7, "ChartQA": 84.5,
    "DocVQA": 91.4, "BLINK": 53.2, "InfoVQA": 74.1, "MMMU": 62.8,
    "LLAVA": 80.3, "ChartQAPro": 70.5, "MMMLU": 76.2,
    "CVQA": 59.8, "CC-OCR": 56.1, "JIC": 70.9, "MulIm-VQA": 63.4,
    "JMMMU": 50.7, "JDocQA": 69.8, "JVB-ItW": 76.5, "VG-VQA": 72.1,
    "Heron": 67.3, "MECHA": 52.9,
  }),
  makeEntry("gpt-4o-2024-11-20", "GPT-4o", {
    "OK-VQA": 71.2, "TextVQA": 78.9, "AI2D": 85.3, "ChartQA": 82.1,
    "DocVQA": 89.7, "BLINK": 58.4, "InfoVQA": 72.8, "MMMU": 68.9,
    "LLAVA": 85.1, "ChartQAPro": 74.3, "MMMLU": 82.1,
    "CVQA": 55.6, "CC-OCR": 52.4, "JIC": 68.7, "MulIm-VQA": 71.2,
    "JMMMU": 56.3, "JDocQA": 65.2, "JVB-ItW": 74.1, "VG-VQA": 70.8,
    "Heron": 63.5, "MECHA": 58.1,
  }),
  makeEntry("Qwen/Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-32B", {
    "OK-VQA": 64.7, "TextVQA": 80.5, "AI2D": 84.2, "ChartQA": 81.8,
    "DocVQA": 88.3, "BLINK": 51.9, "InfoVQA": 71.5, "MMMU": 58.6,
    "LLAVA": 78.4, "ChartQAPro": 67.2, "MMMLU": 73.8,
    "CVQA": 57.3, "CC-OCR": 54.7, "JIC": 68.1, "MulIm-VQA": 60.5,
    "JMMMU": 47.8, "JDocQA": 66.9, "JVB-ItW": 73.2, "VG-VQA": 69.4,
    "Heron": 64.8, "MECHA": 50.3,
  }),
  makeEntry("meta-llama/Llama-3.2-90B-Vision-Instruct", "Llama-3.2-90B", {
    "OK-VQA": 62.5, "TextVQA": 76.8, "AI2D": 80.1, "ChartQA": 78.4,
    "DocVQA": 84.6, "BLINK": 49.3, "InfoVQA": 68.2, "MMMU": 55.1,
    "LLAVA": 75.9, "ChartQAPro": 63.4, "MMMLU": 70.2,
    "CVQA": 53.1, "CC-OCR": 48.6, "JIC": 62.4, "MulIm-VQA": 55.7,
    "JMMMU": 43.2, "JDocQA": 60.1, "JVB-ItW": 68.5, "VG-VQA": 64.3,
    "Heron": 58.7, "MECHA": 45.6,
  }),
  makeEntry("OpenGVLab/InternVL3-38B", "InternVL3-38B", {
    "OK-VQA": 61.8, "TextVQA": 78.2, "AI2D": 82.4, "ChartQA": 80.1,
    "DocVQA": 86.9, "BLINK": 50.5, "InfoVQA": 70.3, "MMMU": 57.2,
    "LLAVA": 76.8, "ChartQAPro": 65.1, "MMMLU": 72.4,
    "CVQA": 55.9, "CC-OCR": 52.3, "JIC": 66.7, "MulIm-VQA": 58.9,
    "JMMMU": 46.1, "JDocQA": 64.5, "JVB-ItW": 71.8, "VG-VQA": 67.2,
    "Heron": 62.4, "MECHA": 48.7,
  }),
  makeEntry("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B", {
    "OK-VQA": 58.3, "TextVQA": 74.1, "AI2D": 78.6, "ChartQA": 75.3,
    "DocVQA": 82.1, "BLINK": 46.8, "InfoVQA": 65.4, "MMMU": 51.3,
    "LLAVA": 72.5, "ChartQAPro": 59.8, "MMMLU": 67.1,
    "CVQA": 50.2, "CC-OCR": 47.1, "JIC": 61.3, "MulIm-VQA": 53.6,
    "JMMMU": 40.5, "JDocQA": 58.7, "JVB-ItW": 65.9, "VG-VQA": 61.5,
    "Heron": 56.2, "MECHA": 42.8,
  }),
  makeEntry("OpenGVLab/InternVL3-8B", "InternVL3-8B", {
    "OK-VQA": 56.7, "TextVQA": 72.3, "AI2D": 76.8, "ChartQA": 73.6,
    "DocVQA": 80.4, "BLINK": 45.1, "InfoVQA": 63.7, "MMMU": 49.8,
    "LLAVA": 70.2, "ChartQAPro": 57.4, "MMMLU": 65.3,
    "CVQA": 48.5, "CC-OCR": 45.3, "JIC": 59.6, "MulIm-VQA": 51.2,
    "JMMMU": 38.9, "JDocQA": 56.4, "JVB-ItW": 63.7, "VG-VQA": 59.8,
    "Heron": 54.1, "MECHA": 40.6,
  }),
  makeEntry("google/gemma-3-27b-it", "Gemma-3-27B", {
    "OK-VQA": 55.1, "TextVQA": 70.8, "AI2D": 75.2, "ChartQA": 71.9,
    "DocVQA": 78.5, "BLINK": 43.7, "InfoVQA": 61.2, "MMMU": 48.1,
    "LLAVA": 68.4, "ChartQAPro": 55.6, "MMMLU": 63.7,
    "CVQA": 46.8, "CC-OCR": 43.5, "JIC": 57.2, "MulIm-VQA": 49.4,
    "JMMMU": 36.7, "JDocQA": 54.1, "JVB-ItW": 61.3, "VG-VQA": 57.5,
    "Heron": 52.3, "MECHA": 38.9,
  }),
  makeEntry("llm-jp/llm-jp-3-vila-14b", "LLM-JP-3-ViLA-14B", {
    "OK-VQA": 42.3, "TextVQA": 58.6, "AI2D": 62.1, "ChartQA": 55.8,
    "DocVQA": 64.2, "BLINK": 38.4, "InfoVQA": 48.7, "MMMU": 35.6,
    "LLAVA": 55.3, "ChartQAPro": 41.2, "MMMLU": 50.4,
    "CVQA": 52.4, "CC-OCR": 49.8, "JIC": 63.5, "MulIm-VQA": 56.1,
    "JMMMU": 42.3, "JDocQA": 62.7, "JVB-ItW": 68.4, "VG-VQA": 65.2,
    "Heron": 61.8, "MECHA": 46.5,
  }),
];

// ── Metadata ────────────────────────────────────────────────────

export const LAST_UPDATED = "2026-04-01";
