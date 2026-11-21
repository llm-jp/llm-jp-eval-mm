"use client";

import { useEffect, useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import { withBasePath } from "@/lib/base-path";

type ScoreMap = Record<string, Record<string, number>>;
type Row = { model: string; url?: string; scores: ScoreMap };
type DatasetUrl = Record<string, { url: string }>;
type DefaultMetrics = { default_metrics: Record<string, string> };
type SortKey = { dataset: string; metric: string; direction: "asc" | "desc" };

export function StaticLeaderboard() {
  const [rows, setRows] = useState<Row[]>([]);
  const [datasetUrl, setDatasetUrl] = useState<DatasetUrl>({});
  const [defaultMetrics, setDefaultMetrics] = useState<Record<string, string>>({});
  const [sort, setSort] = useState<SortKey | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [lb, dm, du] = await Promise.all([
          fetch(withBasePath("/leaderboard.json")).then((r) => r.json()),
          fetch(withBasePath("/default_metrics.json")).then((r) => r.json()),
          fetch(withBasePath("/dataset_url.json")).then((r) => r.json()),
        ]);
        setRows(lb as Row[]);
        setDefaultMetrics((dm as DefaultMetrics).default_metrics);
        setDatasetUrl(du as DatasetUrl);
      } catch (e) {
        console.error(e);
        setError("Failed to load leaderboard data.");
      }
    };
    load();
  }, []);

  const { datasets, metricsByDataset } = useMemo(() => {
    const datasetSet = new Set<string>();
    for (const row of rows) {
      for (const ds of Object.keys(row.scores)) datasetSet.add(ds);
    }
    const datasets = Array.from(datasetSet);
    const metricsByDataset: Record<string, string[]> = {};
    for (const ds of datasets) {
      const mset = new Set<string>();
      for (const row of rows) {
        for (const m of Object.keys(row.scores[ds] ?? {})) mset.add(m);
      }
      metricsByDataset[ds] = Array.from(mset);
    }
    return { datasets, metricsByDataset };
  }, [rows]);

  const sortedRows = useMemo(() => {
    if (!sort) return rows;
    const { dataset, metric, direction } = sort;
    return [...rows].sort((a, b) => {
      const av = a.scores[dataset]?.[metric] ?? -Infinity;
      const bv = b.scores[dataset]?.[metric] ?? -Infinity;
      if (av === bv) return 0;
      const cmp = av < bv ? -1 : 1;
      return direction === "asc" ? cmp : -cmp;
    });
  }, [rows, sort]);

  const handleSort = (dataset: string, metric: string) => {
    setSort((prev) => {
      if (prev?.dataset === dataset && prev.metric === metric) {
        return { dataset, metric, direction: prev.direction === "asc" ? "desc" : "asc" };
      }
      return { dataset, metric, direction: "desc" };
    });
  };

  const sortArrow = (dataset: string, metric: string) => {
    if (sort?.dataset !== dataset || sort.metric !== metric) return "↕";
    return sort.direction === "asc" ? "↑" : "↓";
  };

  return (
    <section className="flex flex-col gap-4">
      <h2 className="text-2xl font-semibold tracking-tight">Leaderboard</h2>
      {error && <p className="text-sm text-destructive">{error}</p>}
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full border-collapse text-sm">
          <thead className="bg-muted">
            <tr>
              <th rowSpan={2} className="sticky left-0 z-10 border-b bg-muted px-3 py-2 text-left align-bottom">
                Model
              </th>
              {datasets.map((ds) => (
                <th
                  key={ds}
                  colSpan={metricsByDataset[ds]?.length || 1}
                  className="border-b border-l px-3 py-2 text-center"
                >
                  {datasetUrl[ds]?.url ? (
                    <a href={datasetUrl[ds].url} target="_blank" rel="noreferrer noopener" className="underline-offset-4 hover:underline">
                      {ds}
                    </a>
                  ) : (
                    ds
                  )}
                </th>
              ))}
            </tr>
            <tr>
              {datasets.map((ds) =>
                metricsByDataset[ds]?.map((m) => (
                  <th
                    key={`${ds}-${m}`}
                    onClick={() => handleSort(ds, m)}
                    className={cn(
                      "cursor-pointer border-b border-l px-3 py-1.5 text-xs font-medium",
                      defaultMetrics[ds] === m && "bg-accent/40",
                    )}
                  >
                    {m} <span className="text-muted-foreground">{sortArrow(ds, m)}</span>
                  </th>
                )),
              )}
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row) => (
              <tr key={row.model} className="even:bg-muted/30">
                <td className="sticky left-0 z-10 border-t bg-background px-3 py-1.5 font-mono text-xs even:bg-muted/30">
                  {row.url ? (
                    <a href={row.url} target="_blank" rel="noreferrer noopener" className="underline-offset-4 hover:underline">
                      {row.model}
                    </a>
                  ) : (
                    row.model
                  )}
                </td>
                {datasets.map((ds) =>
                  metricsByDataset[ds]?.map((m) => {
                    const v = row.scores[ds]?.[m];
                    return (
                      <td
                        key={`${ds}-${m}`}
                        className={cn(
                          "border-t border-l px-3 py-1.5 text-right tabular-nums",
                          defaultMetrics[ds] === m && "bg-accent/20",
                        )}
                      >
                        {typeof v === "number" ? v.toFixed(2) : "—"}
                      </td>
                    );
                  }),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
