"use client";

import { useState, useEffect } from "react";
import { fetchRunResults, type RunResultEntry } from "@/lib/api";
import { cn } from "@/lib/utils";

type Status = "pass" | "fail" | "running" | "pending";

const STATUS_ICON: Record<Status, string> = {
  pass: "\u2705",
  fail: "\u274C",
  running: "\u23F3",
  pending: "\u2B1C",
};

const STATUS_LABEL: Record<Status, string> = {
  pass: "Passed",
  fail: "Failed",
  running: "Running",
  pending: "Pending",
};

const POLL_INTERVAL = 10_000;

export function ResultsMatrix() {
  const [results, setResults] = useState<RunResultEntry[]>([]);

  useEffect(() => {
    const poll = async () => {
      const r = await fetchRunResults();
      setResults(r);
    };
    poll();
    const id = setInterval(poll, POLL_INTERVAL);
    return () => clearInterval(id);
  }, []);

  // Derive unique tasks and models from results
  const tasks = [...new Set(results.map((r) => r.task))];
  const models = [...new Set(results.map((r) => r.model))];

  function getStatus(model: string, task: string): Status {
    const r = results.find((r) => r.model === model && r.task === task);
    return (r?.status as Status) ?? "pending";
  }

  function shortName(name: string): string {
    // "org/model-name" -> "model-name"
    const parts = name.split("/");
    const last = parts[parts.length - 1];
    return last.length > 22 ? last.slice(0, 22) + "\u2026" : last;
  }

  if (results.length === 0) {
    return (
      <div className="rounded-lg border border-[#362d59] bg-[#150f23] p-4">
        <h2 className="text-sm font-semibold text-white mb-3">Results</h2>
        <p className="text-sm text-[#e5e7eb]">
          No results yet. Results will appear as eval.sh runs.
        </p>
      </div>
    );
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
                  key={t}
                  className="text-center text-[#e5e7eb] font-medium pb-2 px-1"
                  title={t}
                >
                  <span className="block max-w-[56px] truncate mx-auto">
                    {t}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model} className="border-t border-[#362d59]/40">
                <td
                  className="py-1.5 pr-2 text-white font-mono truncate max-w-[160px]"
                  title={model}
                >
                  {shortName(model)}
                </td>
                {tasks.map((task) => {
                  const status = getStatus(model, task);
                  return (
                    <td
                      key={task}
                      className="text-center py-1.5 px-1"
                      title={`${model} / ${task}: ${STATUS_LABEL[status]}`}
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
        {(Object.entries(STATUS_ICON) as [Status, string][]).map(
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
