"use client";

import { useState } from "react";
import { BrowserSelector } from "@/components/browser-selector";
import { SampleViewer } from "@/components/sample-viewer";
import { PredictionPanel } from "@/components/prediction-panel";
import { Button } from "@/components/ui/button";
import { MOCK_SAMPLES, TASKS, MODELS } from "@/lib/mock-predictions";
import { ChevronLeft, ChevronRight } from "lucide-react";

export default function BrowserPage() {
  const [selectedTask, setSelectedTask] = useState(TASKS[0].id);
  const [selectedModels, setSelectedModels] = useState([
    MODELS[0].id,
    MODELS[1].id,
  ]);
  const [sampleIndex, setSampleIndex] = useState(0);

  const samples = MOCK_SAMPLES;
  const currentSample = samples[sampleIndex];
  const totalSamples = samples.length;

  const task = TASKS.find((t) => t.id === selectedTask);

  const goToPrevious = () => {
    setSampleIndex((prev) => (prev > 0 ? prev - 1 : totalSamples - 1));
  };

  const goToNext = () => {
    setSampleIndex((prev) => (prev < totalSamples - 1 ? prev + 1 : 0));
  };

  return (
    <div
      className="min-h-screen"
      style={{ backgroundColor: "#f5f4ed" }}
    >
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Page header */}
        <div className="mb-6">
          <h1
            className="text-3xl font-bold tracking-tight"
            style={{ color: "#141413" }}
          >
            Prediction Browser
          </h1>
          <p className="mt-1 text-sm" style={{ color: "#5e5d59" }}>
            Browse individual model predictions and compare ground-truth
            answers across tasks.
          </p>
        </div>

        {/* Selector bar */}
        <div className="mb-6">
          <BrowserSelector
            selectedTask={selectedTask}
            onTaskChange={(taskId) => {
              setSelectedTask(taskId);
              setSampleIndex(0);
            }}
            selectedModels={selectedModels}
            onModelsChange={setSelectedModels}
          />
        </div>

        {/* Task info */}
        {task && (
          <div className="mb-4 flex items-center gap-2">
            <span
              className="rounded-md px-2 py-0.5 text-xs font-medium"
              style={{ backgroundColor: "#e8e6dc", color: "#5e5d59" }}
            >
              {task.name}
            </span>
            <span className="text-xs" style={{ color: "#87867f" }}>
              {task.sampleCount} samples in dataset
            </span>
          </div>
        )}

        {/* Sample viewer */}
        {currentSample && (
          <>
            <div className="mb-6">
              <SampleViewer sample={currentSample} />
            </div>

            {/* Prediction panel */}
            <div className="mb-6">
              <PredictionPanel
                predictions={currentSample.predictions}
                selectedModels={selectedModels}
              />
            </div>
          </>
        )}

        {/* Navigation */}
        <div className="flex items-center justify-center gap-4">
          <Button
            variant="outline"
            size="icon"
            onClick={goToPrevious}
            style={{ borderColor: "#e8e6dc", color: "#5e5d59" }}
          >
            <ChevronLeft className="size-4" />
          </Button>

          <span
            className="min-w-[80px] text-center text-sm font-medium tabular-nums"
            style={{ color: "#141413" }}
          >
            {sampleIndex + 1}{" "}
            <span style={{ color: "#87867f" }}>/ {totalSamples}</span>
          </span>

          <Button
            variant="outline"
            size="icon"
            onClick={goToNext}
            style={{ borderColor: "#e8e6dc", color: "#5e5d59" }}
          >
            <ChevronRight className="size-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
