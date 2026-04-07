"use client";

import { MOCK_PROGRESS } from "@/lib/mock-runs";
import { Timer } from "lucide-react";

function formatEta(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

export function RunProgress() {
  const { taskName, progress, etaSeconds, logs } = MOCK_PROGRESS;

  return (
    <div className="rounded-lg border border-[#362d59] bg-[#150f23] p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-white">Progress</h2>
        <div className="flex items-center gap-1.5 text-xs text-[#e5e7eb]">
          <Timer className="size-3.5" />
          <span>ETA {formatEta(etaSeconds)}</span>
        </div>
      </div>

      {/* Current task + percentage */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-white">
          Running: <span className="font-mono font-semibold">{taskName}</span>
        </span>
        <span className="text-xs font-mono text-[#c2ef4e]">{progress}%</span>
      </div>

      {/* Progress bar */}
      <div className="h-2 rounded-full bg-[#362d59] mb-4 overflow-hidden">
        <div
          className="h-full rounded-full bg-[#c2ef4e] transition-all"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Log output */}
      <div className="rounded-md bg-[#0d0919] border border-[#362d59]/60 p-3 max-h-44 overflow-y-auto">
        <pre className="text-[11px] leading-relaxed font-mono text-[#e5e7eb] whitespace-pre-wrap">
          {logs.map((line, i) => (
            <span key={i} className="block">
              {line}
            </span>
          ))}
          <span className="text-[#c2ef4e] animate-pulse">_</span>
        </pre>
      </div>
    </div>
  );
}
