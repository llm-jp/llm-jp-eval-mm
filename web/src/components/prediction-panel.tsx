"use client";

import { cn } from "@/lib/utils";
import { CheckCircle2, XCircle, Loader2 } from "lucide-react";

/** Unified prediction shape for display. */
export interface PredictionEntry {
  modelId: string;
  modelName: string;
  prediction: string;
  score: number; // 1 = correct, 0 = incorrect, -1 = unknown
}

interface PredictionPanelProps {
  predictions: PredictionEntry[];
  selectedModels: string[];
  loading?: boolean;
}

export function PredictionPanel({
  predictions,
  selectedModels,
  loading,
}: PredictionPanelProps) {
  const filtered = predictions.filter((p) => selectedModels.includes(p.modelId));
  const isComparison = filtered.length > 1;

  if (loading) {
    return (
      <div
        className="flex items-center justify-center rounded-xl border py-12"
        style={{ borderColor: "#e8e6dc", backgroundColor: "#faf9f5" }}
      >
        <Loader2 className="size-5 animate-spin" style={{ color: "#c96442" }} />
        <span className="ml-2 text-sm" style={{ color: "#5e5d59" }}>
          Loading predictions...
        </span>
      </div>
    );
  }

  if (filtered.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-xl border py-12"
        style={{ borderColor: "#e8e6dc", backgroundColor: "#faf9f5" }}
      >
        <span className="text-sm" style={{ color: "#87867f" }}>
          No predictions available for the selected models.
        </span>
      </div>
    );
  }

  return (
    <div
      className="rounded-xl border"
      style={{ borderColor: "#e8e6dc", backgroundColor: "#faf9f5" }}
    >
      <div className="px-4 py-3" style={{ borderBottom: "1px solid #e8e6dc" }}>
        <h3
          className="text-xs font-semibold uppercase tracking-wider"
          style={{ color: "#87867f" }}
        >
          {isComparison ? "Model Comparison" : "Prediction"}
        </h3>
      </div>

      <div
        className={cn(
          "divide-y",
          isComparison ? "grid sm:grid-cols-2 sm:divide-y-0 sm:divide-x" : ""
        )}
        style={
          isComparison
            ? { ["--tw-divide-color" as string]: "#e8e6dc" }
            : { ["--tw-divide-color" as string]: "#e8e6dc" }
        }
      >
        {filtered.map((pred) => (
          <div key={pred.modelId} className="px-4 py-3">
            <div className="mb-1.5 flex items-center gap-2">
              <span
                className="text-sm font-semibold"
                style={{ color: "#141413" }}
              >
                {pred.modelName}
              </span>
              {pred.score === 1 ? (
                <span
                  className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium"
                  style={{ backgroundColor: "#e6f4e6", color: "#2d7a2d" }}
                >
                  <CheckCircle2 className="size-3" />
                  Correct
                </span>
              ) : pred.score === 0 ? (
                <span
                  className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium"
                  style={{ backgroundColor: "#fce8e0", color: "#c96442" }}
                >
                  <XCircle className="size-3" />
                  Incorrect
                </span>
              ) : (
                <span
                  className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium"
                  style={{ backgroundColor: "#e8e6dc", color: "#87867f" }}
                >
                  N/A
                </span>
              )}
            </div>
            <p
              className="text-sm leading-relaxed"
              style={{
                fontFamily: "Georgia, 'Times New Roman', serif",
                color: "#5e5d59",
              }}
            >
              {pred.prediction}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
