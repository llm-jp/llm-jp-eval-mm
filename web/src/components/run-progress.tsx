"use client";

import { type RunStatusData } from "@/lib/api";
import {
  Timer,
  CheckCircle,
  XCircle,
  Cpu,
  Play,
  CircleDashed,
} from "lucide-react";

function formatEta(seconds: number): string {
  if (seconds <= 0) return "--";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${s}s`;
}

function formatElapsed(seconds: number): string {
  if (!seconds) return "0s";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

const PHASE_LABEL: Record<string, string> = {
  loading_model: "Loading model\u2026",
  loading_dataset: "Loading dataset\u2026",
  inferring: "Inferring",
};

export function RunProgress({ status }: { status: RunStatusData }) {
  const progress = status.progress ?? 0;
  const completed = status.completed ?? 0;
  const failed = status.failed ?? 0;
  const total = status.total ?? 0;
  const passed = completed - failed;
  const inference = status.inference;

  return (
    <div className="rounded-lg border border-runner-border bg-runner-surface p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-runner-text">Progress</h2>
        <div className="flex items-center gap-3 text-xs text-runner-text-secondary">
          {status.running && status.etaSeconds != null && (
            <span className="flex items-center gap-1.5">
              <Timer className="size-3.5" />
              ETA {formatEta(status.etaSeconds)}
            </span>
          )}
          {status.elapsedSeconds != null && status.elapsedSeconds > 0 && (
            <span>Elapsed: {formatElapsed(status.elapsedSeconds)}</span>
          )}
        </div>
      </div>

      {!status.running && total === 0 ? (
        <div className="flex flex-col items-center justify-center py-6 text-center">
          <div className="mb-3 rounded-full bg-runner-border/30 p-3">
            <Play className="size-6 text-runner-text-secondary/60" />
          </div>
          <p className="text-sm text-runner-text mb-1">No Evaluation Running</p>
          <p className="text-xs text-runner-text-secondary max-w-[280px]">
            Start eval.sh to see real-time progress tracking here.
          </p>
        </div>
      ) : (
        <>
          {/* Current task + percentage */}
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-runner-text">
              {status.running ? (
                <>
                  Running:{" "}
                  <span className="font-mono font-semibold">
                    {status.currentTask}
                  </span>
                  {" / "}
                  <span className="font-mono text-xs text-runner-text-secondary">
                    {status.currentModel}
                  </span>
                </>
              ) : (
                "Evaluation complete"
              )}
            </span>
            <span className="text-sm font-mono font-semibold text-runner-success tabular-nums">
              {progress}%
            </span>
          </div>

          {/* Overall progress bar */}
          <div className="h-3 rounded-full bg-runner-bar-track mb-4 overflow-hidden">
            <div
              className="h-full rounded-full bg-runner-success transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>

          {/* Inference progress (per-dataset) */}
          {status.running && inference && inference.total > 0 && (
            <div className="mb-4 rounded-md bg-runner-bg p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="flex items-center gap-1.5 text-xs text-runner-text-secondary">
                  <Cpu className="size-3.5 text-runner-primary" />
                  {PHASE_LABEL[inference.phase ?? "inferring"] ?? inference.phase}
                </span>
                <span className="text-xs font-mono text-runner-text-secondary tabular-nums">
                  {inference.current} / {inference.total} samples
                </span>
              </div>
              <div className="h-2 rounded-full bg-runner-bar-track overflow-hidden">
                <div
                  className="h-full rounded-full bg-runner-primary transition-all duration-500 ease-out"
                  style={{
                    width: `${Math.round((inference.current / inference.total) * 100)}%`,
                  }}
                />
              </div>
            </div>
          )}

          {/* Phase indicator (model/dataset loading) */}
          {status.running &&
            inference &&
            inference.total === 0 &&
            inference.phase && (
              <div className="mb-4 rounded-md bg-runner-bg p-3">
                <span className="flex items-center gap-1.5 text-xs text-runner-text-secondary">
                  <Cpu className="size-3.5 text-runner-primary animate-pulse" />
                  {PHASE_LABEL[inference.phase] ?? inference.phase}
                </span>
              </div>
            )}

          {/* Stats cards */}
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-md bg-runner-bg p-2.5 text-center">
              <div className="flex items-center justify-center gap-1.5 mb-0.5">
                <CheckCircle className="size-3.5 text-runner-success" />
                <span className="text-xs text-runner-text-secondary">Passed</span>
              </div>
              <span className="text-lg font-mono font-semibold text-runner-text tabular-nums">
                {passed}
              </span>
              <span className="text-xs text-runner-text-secondary"> / {total}</span>
            </div>
            <div className="rounded-md bg-runner-bg p-2.5 text-center">
              <div className="flex items-center justify-center gap-1.5 mb-0.5">
                <XCircle className="size-3.5 text-runner-danger" />
                <span className="text-xs text-runner-text-secondary">Failed</span>
              </div>
              <span className="text-lg font-mono font-semibold text-runner-text tabular-nums">
                {failed}
              </span>
              <span className="text-xs text-runner-text-secondary"> / {total}</span>
            </div>
            <div className="rounded-md bg-runner-bg p-2.5 text-center">
              <div className="flex items-center justify-center gap-1.5 mb-0.5">
                <CircleDashed className="size-3.5 text-runner-text-secondary" />
                <span className="text-xs text-runner-text-secondary">Remaining</span>
              </div>
              <span className="text-lg font-mono font-semibold text-runner-text tabular-nums">
                {total - completed}
              </span>
              <span className="text-xs text-runner-text-secondary"> / {total}</span>
            </div>
          </div>

          {/* Backend indicator */}
          {status.backend && status.running && (
            <div className="mt-3 text-right">
              <span className="font-mono text-[11px] text-runner-primary">
                {status.backend}
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
