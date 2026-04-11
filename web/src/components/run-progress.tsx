"use client";

import { type RunStatusData } from "@/lib/api";
import { Timer, CheckCircle, XCircle, Cpu } from "lucide-react";

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
  const inference = status.inference;

  return (
    <div className="rounded-lg border border-[#362d59] bg-[#150f23] p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-white">Progress</h2>
        <div className="flex items-center gap-3 text-xs text-[#e5e7eb]">
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
        <p className="text-sm text-[#e5e7eb]">
          No evaluation running. Start eval.sh to see progress here.
        </p>
      ) : (
        <>
          {/* Current task + percentage */}
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-white">
              {status.running ? (
                <>
                  Running:{" "}
                  <span className="font-mono font-semibold">
                    {status.currentTask}
                  </span>
                  {" / "}
                  <span className="font-mono text-xs text-[#e5e7eb]">
                    {status.currentModel}
                  </span>
                </>
              ) : (
                "Evaluation complete"
              )}
            </span>
            <span className="text-xs font-mono text-[#c2ef4e]">
              {progress}%
            </span>
          </div>

          {/* Overall progress bar */}
          <div className="h-2 rounded-full bg-[#362d59] mb-3 overflow-hidden">
            <div
              className="h-full rounded-full bg-[#c2ef4e] transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>

          {/* Inference progress (per-dataset) */}
          {status.running && inference && inference.total > 0 && (
            <div className="mb-3 rounded-md bg-[#1f1633] p-2.5">
              <div className="flex items-center justify-between mb-1.5">
                <span className="flex items-center gap-1.5 text-xs text-[#e5e7eb]">
                  <Cpu className="size-3.5 text-[#6a5fc1]" />
                  {PHASE_LABEL[inference.phase ?? "inferring"] ?? inference.phase}
                </span>
                <span className="text-xs font-mono text-[#e5e7eb]">
                  {inference.current} / {inference.total} samples
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-[#362d59] overflow-hidden">
                <div
                  className="h-full rounded-full bg-[#6a5fc1] transition-all"
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
              <div className="mb-3 rounded-md bg-[#1f1633] p-2.5">
                <span className="flex items-center gap-1.5 text-xs text-[#e5e7eb]">
                  <Cpu className="size-3.5 text-[#6a5fc1] animate-pulse" />
                  {PHASE_LABEL[inference.phase] ?? inference.phase}
                </span>
              </div>
            )}

          {/* Stats row */}
          <div className="flex items-center gap-4 text-xs text-[#e5e7eb]">
            <span className="flex items-center gap-1">
              <CheckCircle className="size-3.5 text-[#c2ef4e]" />
              {completed - failed} / {total} passed
            </span>
            {failed > 0 && (
              <span className="flex items-center gap-1">
                <XCircle className="size-3.5 text-[#ea2261]" />
                {failed} failed
              </span>
            )}
            {status.backend && status.running && (
              <span className="ml-auto font-mono text-[11px] text-[#6a5fc1]">
                {status.backend}
              </span>
            )}
          </div>
        </>
      )}
    </div>
  );
}
