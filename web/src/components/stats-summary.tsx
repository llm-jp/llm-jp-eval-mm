"use client";

import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { BarChart3, Layers, CalendarDays, Wifi } from "lucide-react";
import { fetchTasks, fetchModels } from "@/lib/api";

export interface StatsSummaryProps {
  modelCount: number;
  taskCount: number;
  lastUpdated: string;
  /** Override counts from API (set by LeaderboardTable callback). */
  apiModelCount?: number;
  apiTaskCount?: number;
}

interface StatItem {
  label: string;
  value: number | string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
}

function buildStats(props: StatsSummaryProps, live: { models: number; tasks: number } | null): StatItem[] {
  return [
    {
      label: "Models",
      value: live?.models ?? props.apiModelCount ?? props.modelCount,
      icon: BarChart3,
      description: live ? "Evaluated models (live)" : "Evaluated models",
    },
    {
      label: "Tasks",
      value: live?.tasks ?? props.apiTaskCount ?? props.taskCount,
      icon: Layers,
      description: live ? "Benchmark tasks (live)" : "Benchmark tasks",
    },
    {
      label: "Last Updated",
      value: props.lastUpdated,
      icon: CalendarDays,
      description: "Latest evaluation run",
    },
  ];
}

export function StatsSummary(props: StatsSummaryProps) {
  const [liveStats, setLiveStats] = useState<{ models: number; tasks: number } | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadStats() {
      try {
        const [tasks, models] = await Promise.all([
          fetchTasks(),
          fetchModels(),
        ]);
        if (!cancelled) {
          setLiveStats({ models: models.length, tasks: tasks.length });
        }
      } catch {
        // API unavailable — keep mock values
      }
    }

    loadStats();
    return () => { cancelled = true; };
  }, []);

  const items = buildStats(props, liveStats);

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      {items.map((stat) => (
        <Card
          key={stat.label}
          className="border-[#e5edf5] bg-white"
          style={{
            boxShadow:
              "0 2px 12px rgba(50,50,93,0.06), 0 1px 3px rgba(0,0,0,0.03)",
          }}
        >
          <CardContent className="flex items-center gap-4">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-[#f8f6ff]">
              <stat.icon className="size-5 text-[#533afd]" />
            </div>
            <div>
              <p className="text-2xl font-bold tabular-nums text-[#061b31]">
                {stat.value}
              </p>
              <p className="text-xs text-[#64748d]">
                {stat.description}
                {liveStats && stat.label !== "Last Updated" && (
                  <Wifi className="ml-1 inline size-3 text-[#533afd]" />
                )}
              </p>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
