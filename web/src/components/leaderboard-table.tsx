"use client";

import { useState, useMemo, useCallback, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  type ModelEntry,
  type TaskDef,
  EN_TASKS,
  JA_TASKS,
  makeEntryFromScores,
} from "@/lib/mock-data";
import { fetchTasks, fetchAllScores, type ApiTask } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ArrowUp, ArrowDown, ArrowUpDown, Loader2 } from "lucide-react";

// ── Types ────────────────────────────────────────────────────────

type SortKey = "displayName" | "overall" | "enAvg" | "jaAvg" | string;
type SortDir = "asc" | "desc";

interface LeaderboardTableProps {
  data: ModelEntry[];
  /** Callback to notify parent of live stats when API data loads. */
  onApiLoaded?: (info: { modelCount: number; taskCount: number }) => void;
}

// ── Helpers ──────────────────────────────────────────────────────

/** For a given column (by task displayName), find indices of top-1 and top-2 scores. */
function getTopIndices(
  rows: ModelEntry[],
  key: string,
): { top1: number; top2: number } {
  let top1 = -1;
  let top2 = -1;
  let max1 = -Infinity;
  let max2 = -Infinity;

  rows.forEach((row, i) => {
    const val =
      key === "overall"
        ? row.overall
        : key === "enAvg"
          ? row.enAvg
          : key === "jaAvg"
            ? row.jaAvg
            : row.scores[key];
    if (val == null) return;
    if (val > max1) {
      max2 = max1;
      top2 = top1;
      max1 = val;
      top1 = i;
    } else if (val > max2) {
      max2 = val;
      top2 = i;
    }
  });

  return { top1, top2 };
}

// ── Sort icon ────────────────────────────────────────────────────

function SortIcon({
  column,
  sortKey,
  sortDir,
}: {
  column: string;
  sortKey: SortKey;
  sortDir: SortDir;
}) {
  if (sortKey !== column) {
    return <ArrowUpDown className="ml-1 inline size-3 opacity-30" />;
  }
  return sortDir === "asc" ? (
    <ArrowUp className="ml-1 inline size-3" />
  ) : (
    <ArrowDown className="ml-1 inline size-3" />
  );
}

// ── Cluster header ───────────────────────────────────────────────

function ClusterHeader({
  label,
  span,
}: {
  label: string;
  span: number;
}) {
  return (
    <th
      colSpan={span}
      className="border-b border-[#e5edf5] bg-[#f8fafc] px-2 py-1.5 text-center text-xs font-semibold tracking-wide text-[#64748d]"
    >
      {label}
    </th>
  );
}

// ── Score cell ───────────────────────────────────────────────────

function ScoreCell({
  value,
  isTop1,
  isTop2,
}: {
  value: number | null;
  isTop1: boolean;
  isTop2: boolean;
}) {
  if (value == null) {
    return (
      <TableCell className="text-center text-[#64748d]/50">--</TableCell>
    );
  }
  return (
    <TableCell
      className={cn(
        "text-center tabular-nums text-[#061b31]",
        isTop1 && "font-bold",
        isTop2 && !isTop1 && "underline decoration-[#533afd]/40 underline-offset-2",
      )}
    >
      {value.toFixed(1)}
    </TableCell>
  );
}

// ── Main component ───────────────────────────────────────────────

export function LeaderboardTable({ data, onApiLoaded }: LeaderboardTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("overall");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [apiData, setApiData] = useState<ModelEntry[] | null>(null);
  const [apiTasks, setApiTasks] = useState<{ en: TaskDef[]; ja: TaskDef[] } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function loadFromApi() {
      try {
        const [tasks, allScores] = await Promise.all([
          fetchTasks(),
          fetchAllScores(),
        ]);

        if (cancelled) return;

        // Build task definitions from API
        const apiTaskDefs: TaskDef[] = tasks.map((t: ApiTask) => ({
          taskId: t.task_id,
          displayName: t.display_name,
          cluster: t.cluster,
        }));
        const enTasks = apiTaskDefs.filter((t) => t.cluster === "英語");
        const jaTasks = apiTaskDefs.filter((t) => t.cluster !== "英語");

        // Build a map: model_id -> { taskDisplayName -> score }
        const modelScores: Record<string, Record<string, number | null>> = {};
        const taskDisplayMap: Record<string, string> = {};
        for (const t of apiTaskDefs) {
          taskDisplayMap[t.taskId] = t.displayName;
        }

        for (const [taskId, entries] of Object.entries(allScores)) {
          const displayName = taskDisplayMap[taskId];
          if (!displayName) continue;
          for (const entry of entries) {
            if (!modelScores[entry.model_id]) {
              modelScores[entry.model_id] = {};
            }
            // Take the first metric value as the score
            if (entry.metrics && entry.metrics.length > 0) {
              const firstMetric = entry.metrics[0];
              const val = typeof firstMetric === "object"
                ? Object.values(firstMetric)[0]
                : firstMetric;
              if (typeof val === "number") {
                modelScores[entry.model_id][displayName] = val;
              }
            }
          }
        }

        // Convert to ModelEntry[]
        const entries: ModelEntry[] = Object.entries(modelScores).map(
          ([modelId, scores]) => {
            const shortName = modelId.includes("/")
              ? modelId.split("/").pop()!
              : modelId;
            return makeEntryFromScores(modelId, shortName, scores, enTasks, jaTasks);
          },
        );

        if (entries.length > 0 && !cancelled) {
          setApiData(entries);
          setApiTasks({ en: enTasks, ja: jaTasks });
          onApiLoaded?.({ modelCount: entries.length, taskCount: apiTaskDefs.length });
        }
      } catch {
        // API unavailable — fall back to mock data silently
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadFromApi();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Resolve active data: prefer API data, fall back to props (mock)
  const activeData = apiData ?? data;
  const activeEnTasks = apiTasks?.en ?? EN_TASKS;
  const activeJaTasks = apiTasks?.ja ?? JA_TASKS;

  const handleSort = useCallback(
    (key: SortKey) => {
      if (sortKey === key) {
        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      } else {
        setSortKey(key);
        setSortDir("desc");
      }
    },
    [sortKey],
  );

  const sorted = useMemo(() => {
    const rows = [...activeData];
    rows.sort((a, b) => {
      let va: number | null;
      let vb: number | null;

      if (sortKey === "displayName") {
        const cmp = a.displayName.localeCompare(b.displayName);
        return sortDir === "asc" ? cmp : -cmp;
      }

      if (sortKey === "overall") {
        va = a.overall;
        vb = b.overall;
      } else if (sortKey === "enAvg") {
        va = a.enAvg;
        vb = b.enAvg;
      } else if (sortKey === "jaAvg") {
        va = a.jaAvg;
        vb = b.jaAvg;
      } else {
        va = a.scores[sortKey] ?? null;
        vb = b.scores[sortKey] ?? null;
      }

      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return sortDir === "asc" ? va - vb : vb - va;
    });
    return rows;
  }, [activeData, sortKey, sortDir]);

  // Precompute top-1/top-2 per column
  const aggregateKeys = ["overall", "enAvg", "jaAvg"] as const;
  const allTaskNames = [...activeEnTasks, ...activeJaTasks].map((t) => t.displayName);
  const allKeys = [...aggregateKeys, ...allTaskNames];

  const topMap = useMemo(() => {
    const map: Record<string, { top1: number; top2: number }> = {};
    for (const key of allKeys) {
      map[key] = getTopIndices(sorted, key);
    }
    return map;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sorted]);

  // ── Column header helper ─────────────────────────────────────

  function SortableHead({
    column,
    label,
    className,
  }: {
    column: SortKey;
    label: string;
    className?: string;
  }) {
    return (
      <TableHead
        className={cn(
          "cursor-pointer select-none border-[#e5edf5] px-3 text-center text-xs font-semibold text-[#061b31] transition-colors hover:bg-[#f0f4fa]",
          className,
        )}
        onClick={() => handleSort(column)}
      >
        {label}
        <SortIcon column={column} sortKey={sortKey} sortDir={sortDir} />
      </TableHead>
    );
  }

  // ── Task column group renderer ───────────────────────────────

  function renderTaskHeaders(tasks: TaskDef[]) {
    return tasks.map((t) => (
      <SortableHead key={t.taskId} column={t.displayName} label={t.displayName} />
    ));
  }

  function renderTaskCells(row: ModelEntry, rowIdx: number, tasks: TaskDef[]) {
    return tasks.map((t) => {
      const tops = topMap[t.displayName];
      return (
        <ScoreCell
          key={t.taskId}
          value={row.scores[t.displayName] ?? null}
          isTop1={tops?.top1 === rowIdx}
          isTop2={tops?.top2 === rowIdx}
        />
      );
    });
  }

  // ── Loading state ────────────────────────────────────────────

  if (loading) {
    return (
      <div
        className="flex items-center justify-center rounded-xl border border-[#e5edf5] py-20"
        style={{ boxShadow: "0 4px 24px rgba(50,50,93,0.08), 0 1px 3px rgba(0,0,0,0.04)" }}
      >
        <Loader2 className="mr-2 size-5 animate-spin text-[#533afd]" />
        <span className="text-sm text-[#64748d]">Loading leaderboard data...</span>
      </div>
    );
  }

  return (
    <div
      className="overflow-hidden rounded-xl border border-[#e5edf5]"
      style={{ boxShadow: "0 4px 24px rgba(50,50,93,0.08), 0 1px 3px rgba(0,0,0,0.04)" }}
    >
      {/* Data source indicator */}
      {apiData && (
        <div className="border-b border-[#e5edf5] bg-[#f8f6ff] px-4 py-1.5">
          <span className="text-xs text-[#533afd]">Live data from API</span>
        </div>
      )}
      <div className="overflow-x-auto">
        <Table className="min-w-[1400px]">
          {/* ── Cluster row ─────────────────────────────────── */}
          <thead>
            <tr>
              {/* Sticky model column + 3 aggregate columns */}
              <th
                colSpan={4}
                className="border-b border-[#e5edf5] bg-white px-2 py-1.5"
              />
              <ClusterHeader label="English" span={activeEnTasks.length} />
              <ClusterHeader label="Japanese" span={activeJaTasks.length} />
            </tr>
          </thead>

          {/* ── Sort headers ────────────────────────────────── */}
          <TableHeader className="bg-white">
            <TableRow className="border-[#e5edf5] hover:bg-transparent">
              <TableHead
                className="sticky left-0 z-10 min-w-[180px] cursor-pointer select-none border-r border-[#e5edf5] bg-white px-4 text-left text-xs font-semibold text-[#061b31] transition-colors hover:bg-[#f0f4fa]"
                onClick={() => handleSort("displayName")}
              >
                Model
                <SortIcon
                  column="displayName"
                  sortKey={sortKey}
                  sortDir={sortDir}
                />
              </TableHead>
              <SortableHead
                column="overall"
                label="Overall"
                className="bg-[#f8f6ff] font-bold"
              />
              <SortableHead column="enAvg" label="EN Avg" />
              <SortableHead column="jaAvg" label="JA Avg" />
              {renderTaskHeaders(activeEnTasks)}
              {renderTaskHeaders(activeJaTasks)}
            </TableRow>
          </TableHeader>

          {/* ── Body ────────────────────────────────────────── */}
          <TableBody>
            {sorted.map((row, rowIdx) => (
              <TableRow
                key={row.modelId}
                className="border-[#e5edf5] transition-colors hover:bg-[#f8f6ff]/60"
              >
                {/* Sticky model name */}
                <TableCell className="sticky left-0 z-10 border-r border-[#e5edf5] bg-white px-4 font-medium text-[#061b31] group-hover/row:bg-[#f8f6ff]/60">
                  <div className="flex items-center gap-2">
                    <span className="text-xs tabular-nums text-[#64748d]">
                      {rowIdx + 1}
                    </span>
                    <span>{row.displayName}</span>
                  </div>
                </TableCell>

                {/* Aggregate columns */}
                <ScoreCell
                  value={row.overall}
                  isTop1={topMap["overall"]?.top1 === rowIdx}
                  isTop2={topMap["overall"]?.top2 === rowIdx}
                />
                <ScoreCell
                  value={row.enAvg}
                  isTop1={topMap["enAvg"]?.top1 === rowIdx}
                  isTop2={topMap["enAvg"]?.top2 === rowIdx}
                />
                <ScoreCell
                  value={row.jaAvg}
                  isTop1={topMap["jaAvg"]?.top1 === rowIdx}
                  isTop2={topMap["jaAvg"]?.top2 === rowIdx}
                />

                {/* Task columns */}
                {renderTaskCells(row, rowIdx, activeEnTasks)}
                {renderTaskCells(row, rowIdx, activeJaTasks)}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
