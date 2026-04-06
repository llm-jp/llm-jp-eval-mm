export interface ModelOption {
  id: string;
  label: string;
}

export interface TaskOption {
  id: string;
  label: string;
}

export type RunStatus = "pass" | "fail" | "running" | "pending";

export interface RunResult {
  modelId: string;
  taskId: string;
  status: RunStatus;
}

export interface RunProgress {
  taskName: string;
  progress: number; // 0-100
  etaSeconds: number;
  logs: string[];
}

export const MODELS: ModelOption[] = [
  { id: "llm-jp-3-13b", label: "llm-jp-3-13b-instruct" },
  { id: "llm-jp-3-7b", label: "llm-jp-3-7b-instruct" },
  { id: "gpt-4o", label: "gpt-4o-2024-08-06" },
  { id: "qwen2-vl-72b", label: "Qwen2-VL-72B-Instruct" },
];

export const TASKS: TaskOption[] = [
  { id: "jmmmu", label: "JMMMU" },
  { id: "japanese-heron-bench", label: "JA-Heron-Bench" },
  { id: "ja-vlm-bench-in-the-wild", label: "JA-VLM-Bench-ITW" },
  { id: "ja-multi-image-vqa", label: "JA-Multi-Image-VQA" },
  { id: "jdocqa", label: "JDocQA" },
  { id: "mmmu", label: "MMMU" },
];

export const MOCK_RESULTS: RunResult[] = [
  // llm-jp-3-13b
  { modelId: "llm-jp-3-13b", taskId: "jmmmu", status: "pass" },
  { modelId: "llm-jp-3-13b", taskId: "japanese-heron-bench", status: "pass" },
  { modelId: "llm-jp-3-13b", taskId: "ja-vlm-bench-in-the-wild", status: "fail" },
  { modelId: "llm-jp-3-13b", taskId: "ja-multi-image-vqa", status: "running" },
  { modelId: "llm-jp-3-13b", taskId: "jdocqa", status: "pending" },
  { modelId: "llm-jp-3-13b", taskId: "mmmu", status: "pending" },

  // llm-jp-3-7b
  { modelId: "llm-jp-3-7b", taskId: "jmmmu", status: "pass" },
  { modelId: "llm-jp-3-7b", taskId: "japanese-heron-bench", status: "pass" },
  { modelId: "llm-jp-3-7b", taskId: "ja-vlm-bench-in-the-wild", status: "pass" },
  { modelId: "llm-jp-3-7b", taskId: "ja-multi-image-vqa", status: "fail" },
  { modelId: "llm-jp-3-7b", taskId: "jdocqa", status: "pending" },
  { modelId: "llm-jp-3-7b", taskId: "mmmu", status: "pending" },

  // gpt-4o
  { modelId: "gpt-4o", taskId: "jmmmu", status: "pass" },
  { modelId: "gpt-4o", taskId: "japanese-heron-bench", status: "pass" },
  { modelId: "gpt-4o", taskId: "ja-vlm-bench-in-the-wild", status: "pass" },
  { modelId: "gpt-4o", taskId: "ja-multi-image-vqa", status: "pass" },
  { modelId: "gpt-4o", taskId: "jdocqa", status: "running" },
  { modelId: "gpt-4o", taskId: "mmmu", status: "pending" },

  // qwen2-vl-72b
  { modelId: "qwen2-vl-72b", taskId: "jmmmu", status: "pending" },
  { modelId: "qwen2-vl-72b", taskId: "japanese-heron-bench", status: "pending" },
  { modelId: "qwen2-vl-72b", taskId: "ja-vlm-bench-in-the-wild", status: "pending" },
  { modelId: "qwen2-vl-72b", taskId: "ja-multi-image-vqa", status: "pending" },
  { modelId: "qwen2-vl-72b", taskId: "jdocqa", status: "pending" },
  { modelId: "qwen2-vl-72b", taskId: "mmmu", status: "pending" },
];

export const MOCK_PROGRESS: RunProgress = {
  taskName: "JMMMU",
  progress: 67,
  etaSeconds: 214,
  logs: [
    "[12:04:01] Initializing vLLM backend...",
    "[12:04:03] Loading model llm-jp-3-13b-instruct (4-bit quantized)",
    "[12:04:18] Model loaded on GPU 0,2 — 34.2 GiB VRAM allocated",
    "[12:04:19] Starting task: JMMMU (820 samples)",
    "[12:04:19] Processing batch 1/5 ...",
    "[12:05:42] Processing batch 2/5 ...",
    "[12:07:11] Processing batch 3/5 ...",
    "[12:08:33] Processing batch 4/5 ...",
  ],
};
