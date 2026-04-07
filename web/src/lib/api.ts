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
