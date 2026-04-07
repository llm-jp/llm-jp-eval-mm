"use client";

import { useState, useEffect } from "react";
import {
  MODELS,
  TASKS,
  MOCK_RESULTS,
  type RunStatus,
  type ModelOption,
  type TaskOption,
  type RunResult,
} from "@/lib/mock-runs";
import { fetchTasks, fetchModels, fetchResults } from "@/lib/api";
import { cn } from "@/lib/utils";

const STATUS_ICON: Record<RunStatus, string> = {
  pass: "\u2705",
  fail: "\u274C",
  running: "\u23F3",
  pending: "\u2B1C",
};

const STATUS_LABEL: Record<RunStatus, string> = {
  pass: "Passed",
  fail: "Failed",
  running: "Running",
  pending: "Pending",
};

/**
 * Map an API result entry to a RunStatus.
 * If the result has metrics it counts as "pass"; otherwise "pending".
 * This heuristic can be refined once the backend exposes richer status info.
 */
function apiResultToStatus(r: {
  metrics: unknown[];
  created_at: string;
}): RunStatus {
  if (r.metrics && r.metrics.length > 0) return "pass";
  return "pending";
}

export function ResultsMatrix() {
  const [models, setModels] = useState<ModelOption[]>(MODELS);
  const [tasks, setTasks] = useState<TaskOption[]>(TASKS);
  const [results, setResults] = useState<RunResult[]>(MOCK_RESULTS);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [apiTasks, apiModels, apiResults] = await Promise.all([
          fetchTasks(),
          fetchModels(),
          fetchResults(),
        ]);

        if (cancelled) return;

        if (apiTasks.length > 0) {
          setTasks(
            apiTasks.map((t) => ({ id: t.task_id, label: t.display_name }))
          );
        }

        if (apiModels.length > 0) {
          setModels(apiModels.map((m) => ({ id: m, label: m })));
        }

        if (apiResults.length > 0) {
          setResults(
            apiResults.map((r) => ({
              modelId: r.model_id,
              taskId: r.task_id,
              status: apiResultToStatus(r),
            }))
          );
        }
      } catch {
        // API unavailable — keep mock data (already set as initial state).
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  function getStatus(modelId: string, taskId: string): RunStatus {
    const r = results.find(
      (r) => r.modelId === modelId && r.taskId === taskId
    );
    return r?.status ?? "pending";
  }

  return (
    <div className="rounded-lg border border-[#362d59] bg-[#150f23] p-4">
      <h2 className="text-sm font-semibold text-white mb-3">Results</h2>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left text-[#e5e7eb] font-medium pb-2 pr-2">
                Model
              </th>
              {tasks.map((t) => (
                <th
                  key={t.id}
                  className="text-center text-[#e5e7eb] font-medium pb-2 px-1"
                  title={t.label}
                >
                  <span className="block max-w-[56px] truncate mx-auto">
                    {t.label}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model.id} className="border-t border-[#362d59]/40">
                <td className="py-1.5 pr-2 text-white font-mono truncate max-w-[120px]">
                  {model.label.length > 18
                    ? model.label.slice(0, 18) + "\u2026"
                    : model.label}
                </td>
                {tasks.map((task) => {
                  const status = getStatus(model.id, task.id);
                  return (
                    <td
                      key={task.id}
                      className="text-center py-1.5 px-1"
                      title={`${model.label} / ${task.label}: ${STATUS_LABEL[status]}`}
                    >
                      <span
                        className={cn(
                          "inline-block text-sm leading-none",
                          status === "running" && "animate-pulse"
                        )}
                      >
                        {STATUS_ICON[status]}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="mt-3 flex flex-wrap gap-3 text-[10px] text-[#e5e7eb]">
        {(Object.entries(STATUS_ICON) as [RunStatus, string][]).map(
          ([status, icon]) => (
            <span key={status} className="flex items-center gap-1">
              <span className="text-sm">{icon}</span>
              {STATUS_LABEL[status]}
            </span>
          )
        )}
      </div>
    </div>
  );
}
