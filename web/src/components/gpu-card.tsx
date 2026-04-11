"use client";

import { type GpuData as GpuInfo } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Cpu, Thermometer, HardDrive } from "lucide-react";

const STATUS_RING: Record<GpuInfo["status"], string> = {
  idle: "border-[#c2ef4e]/40",
  active: "border-[#6a5fc1]",
  high: "border-[#ea2261]",
};

const STATUS_DOT: Record<GpuInfo["status"], string> = {
  idle: "bg-[#c2ef4e]",
  active: "bg-[#6a5fc1]",
  high: "bg-[#ea2261]",
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
  const memPct = Math.round((gpu.memoryUsed / gpu.memoryTotal) * 100);

  return (
    <div
      className={cn(
        "rounded-lg border-2 p-4 transition-colors",
        "bg-[#150f23]",
        STATUS_RING[gpu.status]
      )}
    >
      {/* Header row */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono text-[#e5e7eb]">
          GPU {gpu.id}
        </span>
        <div className="flex items-center gap-1.5">
          <span
            className={cn("inline-block size-2 rounded-full", STATUS_DOT[gpu.status])}
          />
          <span className="text-xs text-[#e5e7eb]">
            {STATUS_LABEL[gpu.status]}
          </span>
        </div>
      </div>

      {/* GPU name */}
      <p className="mt-1 text-sm font-medium text-white truncate">{gpu.name}</p>

      {/* Stats */}
      <div className="mt-3 grid gap-2">
        {/* Utilization */}
        <div className="flex items-center gap-2">
          <Cpu className="size-3.5 text-[#e5e7eb]" />
          <div className="flex-1">
            <div className="h-1.5 rounded-full bg-[#362d59]">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${gpu.utilization}%`,
                  backgroundColor:
                    gpu.status === "high" ? "#ea2261" : "#c2ef4e",
                }}
              />
            </div>
          </div>
          <span className="text-xs font-mono text-white w-9 text-right">
            {gpu.utilization}%
          </span>
        </div>

        {/* Memory */}
        <div className="flex items-center gap-2">
          <HardDrive className="size-3.5 text-[#e5e7eb]" />
          <div className="flex-1">
            <div className="h-1.5 rounded-full bg-[#362d59]">
              <div
                className="h-full rounded-full bg-[#6a5fc1] transition-all"
                style={{ width: `${memPct}%` }}
              />
            </div>
          </div>
          <span className="text-xs font-mono text-white w-9 text-right">
            {memPct}%
          </span>
        </div>

        {/* Memory label + temperature */}
        <div className="flex items-center justify-between text-[11px] text-[#e5e7eb]">
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
