"use client";

import { useState, useEffect } from "react";
import { fetchRunResults, type RunResultEntry } from "@/lib/api";
import { CheckCircle, XCircle, Loader2, Minus, LayoutGrid } from "lucide-react";

type Status = "pass" | "fail" | "running" | "pending";

const STATUS_LABEL: Record<Status, string> = {
  pass: "Passed",
  fail: "Failed",
  running: "Running",
  pending: "Pending",
};

function StatusIcon({ status }: { status: Status }) {
  switch (status) {
    case "pass":
      return <CheckCircle className="size-4 text-runner-success" />;
    case "fail":
      return <XCircle className="size-4 text-runner-danger" />;
    case "running":
      return <Loader2 className="size-4 text-runner-primary animate-spin" />;
    case "pending":
      return <Minus className="size-4 text-runner-text-secondary/40" />;
  }
}

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

  const tasks = [...new Set(results.map((r) => r.task))];
  const models = [...new Set(results.map((r) => r.model))];

  function getStatus(model: string, task: string): Status {
    const entry = results.find((e) => e.model === model && e.task === task);
    return (entry?.status as Status) ?? "pending";
  }

  function shortName(name: string): string {
    const parts = name.split("/");
    const last = parts[parts.length - 1];
    return last.length > 22 ? last.slice(0, 22) + "\u2026" : last;
  }

  if (results.length === 0) {
    return (
      <div className="rounded-lg border border-runner-border bg-runner-surface p-6">
        <div className="flex flex-col items-center justify-center py-6 text-center">
          <div className="mb-3 rounded-full bg-runner-border/30 p-3">
            <LayoutGrid className="size-6 text-runner-text-secondary/60" />
          </div>
          <h2 className="text-sm font-semibold text-runner-text mb-1">
            No Results Yet
          </h2>
          <p className="text-xs text-runner-text-secondary max-w-[240px]">
            Results will appear here as eval.sh processes tasks and models.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-runner-border bg-runner-surface p-4">
      <h2 className="text-sm font-semibold text-runner-text mb-3">Results</h2>

      {/* Scrollable table with fade hint */}
      <div className="relative">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr>
                <th className="sticky left-0 z-10 bg-runner-surface text-left text-runner-text-secondary font-medium pb-2 pr-3">
                  Model
                </th>
                {tasks.map((t) => (
                  <th
                    key={t}
                    className="text-center text-runner-text-secondary font-medium pb-2 px-1.5"
                    title={t}
                  >
                    <span className="block max-w-[64px] truncate mx-auto">
                      {t}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <tr
                  key={model}
                  className="border-t border-runner-border-subtle transition-colors hover:bg-runner-bg/60"
                >
                  <td
                    className="sticky left-0 z-10 bg-runner-surface py-2 pr-3 text-runner-text font-mono truncate max-w-[180px]"
                    title={model}
                  >
                    {shortName(model)}
                  </td>
                  {tasks.map((task) => {
                    const status = getStatus(model, task);
                    return (
                      <td
                        key={task}
                        className="text-center py-2 px-1.5"
                        title={`${model} / ${task}: ${STATUS_LABEL[status]}`}
                      >
                        <span className="inline-flex items-center justify-center">
                          <StatusIcon status={status} />
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-3 flex flex-wrap gap-4 text-[11px] text-runner-text-secondary">
        {(["pass", "fail", "running", "pending"] as Status[]).map((status) => (
          <span key={status} className="flex items-center gap-1.5">
            <StatusIcon status={status} />
            {STATUS_LABEL[status]}
          </span>
        ))}
      </div>
    </div>
  );
}
