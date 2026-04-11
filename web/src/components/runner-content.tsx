"use client";

import { useEffect, useState } from "react";
import { GpuCard } from "@/components/gpu-card";
import { RunProgress } from "@/components/run-progress";
import { ResultsMatrix } from "@/components/results-matrix";
import {
  fetchGpus,
  fetchRunStatus,
  type GpuData,
  type RunStatusData,
} from "@/lib/api";

const POLL_INTERVAL = 5_000;

export default function RunnerContent() {
  const [gpus, setGpus] = useState<GpuData[]>([]);
  const [runStatus, setRunStatus] = useState<RunStatusData>({ running: false });

  useEffect(() => {
    const poll = async () => {
      const [g, s] = await Promise.all([fetchGpus(), fetchRunStatus()]);
      setGpus(g);
      setRunStatus(s);
    };
    poll();
    const id = setInterval(poll, POLL_INTERVAL);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold tracking-tight text-white">
          Runner Dashboard
        </h1>
        <p className="mt-1 text-sm text-[#e5e7eb]">
          Monitor GPUs, configure tasks, and track evaluation progress.
        </p>
      </div>

      <section className="mb-6">
        {gpus.length === 0 ? (
          <p className="text-sm text-[#e5e7eb]">
            No GPU data available. Waiting for nvidia-smi...
          </p>
        ) : (
          <div className="grid gap-3 grid-cols-2 lg:grid-cols-4">
            {gpus.map((gpu) => (
              <GpuCard key={gpu.id} gpu={gpu} />
            ))}
          </div>
        )}
      </section>

      <section className="mb-6">
        <ResultsMatrix />
      </section>

      <section>
        <RunProgress status={runStatus} />
      </section>
    </div>
  );
}
