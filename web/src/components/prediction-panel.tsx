"use client";

import type { ModelPrediction } from "@/lib/mock-predictions";
import { cn } from "@/lib/utils";
import { CheckCircle2, XCircle } from "lucide-react";

interface PredictionPanelProps {
  predictions: ModelPrediction[];
  selectedModels: string[];
}

export function PredictionPanel({
  predictions,
  selectedModels,
}: PredictionPanelProps) {
  const filtered = predictions.filter((p) => selectedModels.includes(p.modelId));
  const isComparison = filtered.length > 1;

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
              ) : (
                <span
                  className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium"
                  style={{ backgroundColor: "#fce8e0", color: "#c96442" }}
                >
                  <XCircle className="size-3" />
                  Incorrect
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
