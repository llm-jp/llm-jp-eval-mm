"use client";

import { type GpuData as GpuInfo } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Cpu, Thermometer, HardDrive } from "lucide-react";

const STATUS_RING: Record<GpuInfo["status"], string> = {
  idle: "border-runner-success/40",
  active: "border-runner-primary",
  high: "border-runner-danger",
};

const STATUS_DOT: Record<GpuInfo["status"], string> = {
  idle: "bg-runner-success",
  active: "bg-runner-primary",
  high: "bg-runner-danger",
};

const STATUS_GLOW: Record<GpuInfo["status"], string> = {
  idle: "",
  active: "shadow-[0_0_12px_var(--runner-glow-primary)]",
  high: "shadow-[0_0_12px_var(--runner-glow-danger)]",
};

const STATUS_LABEL: Record<GpuInfo["status"], string> = {
  idle: "Idle",
  active: "Active",
  high: "High",
};

function formatMemory(mib: number): string {
  if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GiB`;
  return `${mib} MiB`;
}

export function GpuCard({ gpu }: { gpu: GpuInfo }) {
  const memPct =
    gpu.memoryTotal > 0
      ? Math.round((gpu.memoryUsed / gpu.memoryTotal) * 100)
      : 0;

  return (
    <div
      className={cn(
        "rounded-lg border-2 p-4 transition-all duration-300",
        "bg-runner-surface",
        STATUS_RING[gpu.status],
        STATUS_GLOW[gpu.status],
      )}
    >
      {/* Header row */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono text-runner-text-secondary">
          GPU {gpu.id}
        </span>
        <div className="flex items-center gap-1.5">
          <span
            className={cn(
              "inline-block size-2 rounded-full",
              STATUS_DOT[gpu.status],
              gpu.status === "active" && "animate-pulse",
            )}
          />
          <span className="text-xs text-runner-text-secondary">
            {STATUS_LABEL[gpu.status]}
          </span>
        </div>
      </div>

      {/* GPU name */}
      <p
        className="mt-1 text-sm font-medium text-runner-text truncate"
        title={gpu.name}
      >
        {gpu.name}
      </p>

      {/* Stats */}
      <div className="mt-3 grid gap-2.5">
        {/* Utilization */}
        <div className="flex items-center gap-2">
          <Cpu className="size-3.5 shrink-0 text-runner-text-secondary" />
          <div className="flex-1">
            <div className="h-2 rounded-full bg-runner-bar-track overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500 ease-out"
                style={{
                  width: `${gpu.utilization}%`,
                  backgroundColor:
                    gpu.status === "high"
                      ? "var(--runner-danger)"
                      : "var(--runner-success)",
                }}
              />
            </div>
          </div>
          <span className="text-xs font-mono text-runner-text w-9 text-right tabular-nums">
            {gpu.utilization}%
          </span>
        </div>

        {/* Memory */}
        <div className="flex items-center gap-2">
          <HardDrive className="size-3.5 shrink-0 text-runner-text-secondary" />
          <div className="flex-1">
            <div className="h-2 rounded-full bg-runner-bar-track overflow-hidden">
              <div
                className="h-full rounded-full bg-runner-primary transition-all duration-500 ease-out"
                style={{ width: `${memPct}%` }}
              />
            </div>
          </div>
          <span className="text-xs font-mono text-runner-text w-9 text-right tabular-nums">
            {memPct}%
          </span>
        </div>

        {/* Memory label + temperature */}
        <div className="flex items-center justify-between text-[11px] text-runner-text-secondary">
          <span>
            {formatMemory(gpu.memoryUsed)} / {formatMemory(gpu.memoryTotal)}
          </span>
          <span className="flex items-center gap-1">
            <Thermometer className="size-3" />
            {gpu.temperature}&deg;C
          </span>
        </div>
      </div>
    </div>
  );
}
