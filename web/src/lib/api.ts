/**
 * API client for the eval_mm FastAPI backend.
 *
 * When the backend is unreachable (e.g. during static export / GitHub Pages),
 * callers should fall back to mock data.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ApiTask {
  task_id: string;
  display_name: string;
  cluster: string;
}

export interface ApiResult {
  task_id: string;
  model_id: string;
  metrics: unknown[];
  created_at: string;
}

export async function fetchTasks(): Promise<ApiTask[]> {
  const res = await fetch(`${API_BASE}/api/tasks`);
  if (!res.ok) throw new Error(`Failed to fetch tasks: ${res.status}`);
  return res.json();
}

export async function fetchModels(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`);
  return res.json();
}

export async function fetchResults(): Promise<ApiResult[]> {
  const res = await fetch(`${API_BASE}/api/results`);
  if (!res.ok) throw new Error(`Failed to fetch results: ${res.status}`);
  return res.json();
}

export interface ApiScoreEntry {
  model_id: string;
  metrics: Record<string, number>[];
}

export interface ApiScoresResponse {
  task_id: string;
  models: ApiScoreEntry[];
}

export async function fetchScores(taskId: string): Promise<ApiScoresResponse> {
  const res = await fetch(`${API_BASE}/api/scores/${taskId}`);
  if (!res.ok) throw new Error(`Failed to fetch scores for ${taskId}: ${res.status}`);
  return res.json();
}

export async function fetchAllScores(): Promise<Record<string, ApiScoreEntry[]>> {
  const tasks = await fetchTasks();
  const scores: Record<string, ApiScoreEntry[]> = {};
  for (const task of tasks) {
    try {
      const resp = await fetchScores(task.task_id);
      scores[task.task_id] = resp.models;
    } catch {
      /* skip unavailable tasks */
    }
  }
  return scores;
}

// ── Prediction browsing ──────────────────────────────────────────

export interface ApiPrediction {
  question_id: string;
  text: string; // model's prediction
  answer: string; // ground truth
  input_text: string; // the question / prompt
  [key: string]: unknown; // task-specific score fields (e.g. "jmmmu": 1)
}

export interface ApiPredictionsResponse {
  task_id: string;
  model_id: string;
  total: number;
  offset: number;
  limit: number;
  predictions: ApiPrediction[];
}

export async function fetchPredictions(
  taskId: string,
  modelId: string,
  offset = 0,
  limit = 10,
): Promise<ApiPredictionsResponse> {
  const res = await fetch(
    `${API_BASE}/api/predictions/${taskId}/${encodeURIComponent(modelId)}?offset=${offset}&limit=${limit}`,
  );
  if (!res.ok)
    throw new Error(`Failed to fetch predictions: ${res.status}`);
  return res.json();
}

/** Discover which task/model combinations have results. */
export async function fetchAvailableResults(): Promise<ApiResult[]> {
  return fetchResults();
}
