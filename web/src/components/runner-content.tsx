"use client";

import { useEffect, useState, useCallback } from "react";
import { GpuCard } from "@/components/gpu-card";
import { RunProgress } from "@/components/run-progress";
import { ResultsMatrix } from "@/components/results-matrix";
import {
  fetchGpus,
  fetchRunStatus,
  type GpuData,
  type RunStatusData,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { Cpu, MonitorDot } from "lucide-react";

const POLL_INTERVAL = 5_000;

export default function RunnerContent() {
  const [gpus, setGpus] = useState<GpuData[]>([]);
  const [runStatus, setRunStatus] = useState<RunStatusData>({ running: false });
  const [connected, setConnected] = useState<boolean | null>(null);

  const poll = useCallback(async () => {
    try {
      const [g, s] = await Promise.all([fetchGpus(), fetchRunStatus()]);
      setGpus(g);
      setRunStatus(s);
      setConnected(true);
    } catch {
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, POLL_INTERVAL);
    return () => clearInterval(id);
  }, [poll]);

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="mb-8 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-runner-text">
            Runner Dashboard
          </h1>
          <p className="mt-1 text-sm text-runner-text-secondary">
            Monitor GPUs, configure tasks, and track evaluation progress.
          </p>
        </div>
        {/* Connection indicator */}
        {connected !== null && (
          <div
            className={cn(
              "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs",
              connected
                ? "bg-runner-success/10 text-runner-success"
                : "bg-runner-danger/10 text-runner-danger",
            )}
          >
            <span
              className={cn(
                "inline-block size-1.5 rounded-full",
                connected ? "bg-runner-success" : "bg-runner-danger",
              )}
            />
            {connected ? "Connected" : "Disconnected"}
          </div>
        )}
      </div>

      {/* GPU section */}
      <section className="mb-8">
        <div className="mb-3 flex items-center gap-2">
          <Cpu className="size-4 text-runner-text-secondary" />
          <h2 className="text-xs font-semibold uppercase tracking-wider text-runner-text-secondary">
            GPU Status
          </h2>
        </div>
        {gpus.length === 0 ? (
          <div className="rounded-lg border border-runner-border bg-runner-surface p-6">
            <div className="flex flex-col items-center justify-center py-4 text-center">
              <div className="mb-3 rounded-full bg-runner-border/30 p-3">
                <MonitorDot className="size-6 text-runner-text-secondary/60" />
              </div>
              <p className="text-sm text-runner-text mb-1">No GPU Data</p>
              <p className="text-xs text-runner-text-secondary max-w-[260px]">
                Waiting for nvidia-smi telemetry. Make sure the API backend is running.
              </p>
            </div>
          </div>
        ) : (
          <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
            {gpus.map((gpu) => (
              <GpuCard key={gpu.id} gpu={gpu} />
            ))}
          </div>
        )}
      </section>

      {/* Results section */}
      <section className="mb-8">
        <div className="mb-3 flex items-center gap-2">
          <span className="flex size-4 items-center justify-center text-runner-text-secondary">
            <svg
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              className="size-4"
            >
              <rect x="1" y="1" width="6" height="6" rx="1" />
              <rect x="9" y="1" width="6" height="6" rx="1" />
              <rect x="1" y="9" width="6" height="6" rx="1" />
              <rect x="9" y="9" width="6" height="6" rx="1" />
            </svg>
          </span>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-runner-text-secondary">
            Results Matrix
          </h2>
        </div>
        <ResultsMatrix />
      </section>

      {/* Progress section */}
      <section>
        <div className="mb-3 flex items-center gap-2">
          <span className="flex size-4 items-center justify-center text-runner-text-secondary">
            <svg
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              className="size-4"
            >
              <rect x="1" y="6" width="14" height="4" rx="2" />
              <rect x="1" y="6" width="8" height="4" rx="2" fill="currentColor" opacity="0.3" />
            </svg>
          </span>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-runner-text-secondary">
            Evaluation Progress
          </h2>
        </div>
        <RunProgress status={runStatus} />
      </section>
    </div>
  );
}
