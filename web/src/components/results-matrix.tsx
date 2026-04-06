"use client";

import { MODELS, TASKS, MOCK_RESULTS, type RunStatus } from "@/lib/mock-runs";
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

function getStatus(modelId: string, taskId: string): RunStatus {
  const r = MOCK_RESULTS.find(
    (r) => r.modelId === modelId && r.taskId === taskId
  );
  return r?.status ?? "pending";
}

export function ResultsMatrix() {
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
              {TASKS.map((t) => (
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
            {MODELS.map((model) => (
              <tr key={model.id} className="border-t border-[#362d59]/40">
                <td className="py-1.5 pr-2 text-white font-mono truncate max-w-[120px]">
                  {model.label.length > 18
                    ? model.label.slice(0, 18) + "\u2026"
                    : model.label}
                </td>
                {TASKS.map((task) => {
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
