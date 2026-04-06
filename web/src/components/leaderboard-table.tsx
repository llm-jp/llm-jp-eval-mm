"use client";

import { useState, useMemo, useCallback } from "react";
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
} from "@/lib/mock-data";
import { cn } from "@/lib/utils";
import { ArrowUp, ArrowDown, ArrowUpDown } from "lucide-react";

// ── Types ────────────────────────────────────────────────────────

type SortKey = "displayName" | "overall" | "enAvg" | "jaAvg" | string;
type SortDir = "asc" | "desc";

interface LeaderboardTableProps {
  data: ModelEntry[];
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

export function LeaderboardTable({ data }: LeaderboardTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("overall");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

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
    const rows = [...data];
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
  }, [data, sortKey, sortDir]);

  // Precompute top-1/top-2 per column
  const aggregateKeys = ["overall", "enAvg", "jaAvg"] as const;
  const allTaskNames = [...EN_TASKS, ...JA_TASKS].map((t) => t.displayName);
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

  return (
    <div
      className="overflow-hidden rounded-xl border border-[#e5edf5]"
      style={{ boxShadow: "0 4px 24px rgba(50,50,93,0.08), 0 1px 3px rgba(0,0,0,0.04)" }}
    >
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
              <ClusterHeader label="English" span={EN_TASKS.length} />
              <ClusterHeader label="Japanese" span={JA_TASKS.length} />
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
              {renderTaskHeaders(EN_TASKS)}
              {renderTaskHeaders(JA_TASKS)}
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
                {renderTaskCells(row, rowIdx, EN_TASKS)}
                {renderTaskCells(row, rowIdx, JA_TASKS)}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
