"use client";

import { ImageIcon } from "lucide-react";

/** Unified sample shape used across mock and real data. */
export interface SampleData {
  id: string;
  imageUrl: string | null;
  placeholderColor: string;
  question: string;
  groundTruth: string;
}

interface SampleViewerProps {
  sample: SampleData;
}

export function SampleViewer({ sample }: SampleViewerProps) {
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Image area */}
      <div
        className="flex aspect-[4/3] items-center justify-center overflow-hidden rounded-xl border"
        style={{
          borderColor: "#e8e6dc",
          backgroundColor: sample.imageUrl ? undefined : sample.placeholderColor,
        }}
      >
        {sample.imageUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={sample.imageUrl}
            alt="Sample image"
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="flex flex-col items-center gap-3 text-center">
            <ImageIcon
              className="size-12 opacity-40"
              style={{ color: "#5e5d59" }}
            />
            <span
              className="text-sm font-medium opacity-60"
              style={{ color: "#5e5d59" }}
            >
              {sample.id}
            </span>
          </div>
        )}
      </div>

      {/* Question + Ground truth */}
      <div className="flex flex-col gap-4">
        <div>
          <h3
            className="mb-2 text-xs font-semibold uppercase tracking-wider"
            style={{ color: "#87867f" }}
          >
            Question
          </h3>
          <p
            className="text-base leading-relaxed"
            style={{
              fontFamily: "Georgia, 'Times New Roman', serif",
              color: "#141413",
            }}
          >
            {sample.question}
          </p>
        </div>

        <div
          className="rounded-lg p-4"
          style={{ backgroundColor: "#141413" }}
        >
          <h3
            className="mb-1.5 text-xs font-semibold uppercase tracking-wider"
            style={{ color: "#87867f" }}
          >
            Ground Truth
          </h3>
          <p
            className="text-sm font-medium leading-relaxed"
            style={{
              fontFamily: "Georgia, 'Times New Roman', serif",
              color: "#faf9f5",
            }}
          >
            {sample.groundTruth}
          </p>
        </div>
      </div>
    </div>
  );
}
