"use client";

import { useCallback, useEffect, useState } from "react";
import { BrowserSelector } from "@/components/browser-selector";
import { SampleViewer, type SampleData } from "@/components/sample-viewer";
import { PredictionPanel, type PredictionEntry } from "@/components/prediction-panel";
import { Button } from "@/components/ui/button";
import {
  MOCK_SAMPLES,
  TASKS as MOCK_TASKS,
  MODELS as MOCK_MODELS,
} from "@/lib/mock-predictions";
import {
  fetchTasks,
  fetchModels,
  fetchPredictions,
  fetchResults,
  type ApiTask,
  type ApiPrediction,
  type ApiResult,
} from "@/lib/api";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { IS_STATIC_EXPORT } from "@/lib/base-path";
import { BackendOnlyNotice } from "@/components/backend-only-notice";

// ── Helpers ──────────────────────────────────────────────────────

/** Deterministic placeholder color from a string. */
function placeholderColorFor(id: string): string {
  const palette = ["#d4c5a9", "#b8c5d4", "#c5d4b8", "#d4b8c5", "#c5b8d4"];
  let hash = 0;
  for (let i = 0; i < id.length; i++) {
    hash = (hash * 31 + id.charCodeAt(i)) | 0;
  }
  return palette[Math.abs(hash) % palette.length];
}

/** Derive the per-sample score from a prediction record.
 *
 * The API stores a score under a key matching the task_id (e.g. "jmmmu": 1).
 * If the task-specific key is missing, try comparing text vs answer.
 */
function extractScore(pred: ApiPrediction, taskId: string): number {
  if (taskId in pred) {
    const v = pred[taskId];
    return typeof v === "number" ? v : -1;
  }
  // Fallback: exact match comparison
  if (pred.text && pred.answer) {
    return pred.text.trim() === pred.answer.trim() ? 1 : 0;
  }
  return -1;
}

/** Short display name from a model id like "Qwen/Qwen2.5-VL-3B-Instruct". */
function modelDisplayName(modelId: string): string {
  return modelId.split("/").pop() ?? modelId;
}

// ── Types for internal state ─────────────────────────────────────

interface TaskOption {
  id: string;
  name: string;
  sampleCount?: number;
}

interface ModelOption {
  id: string;
  name: string;
}

// ── Page component ───────────────────────────────────────────────

export default function BrowserPage() {
  if (IS_STATIC_EXPORT) return <BackendOnlyNotice page="Browser" />;
  return <BrowserPageInner />;
}

function BrowserPageInner() {
  // Data sources
  const [tasks, setTasks] = useState<TaskOption[]>(
    MOCK_TASKS.map((t) => ({ id: t.id, name: t.name, sampleCount: t.sampleCount })),
  );
  const [models, setModels] = useState<ModelOption[]>(
    MOCK_MODELS.map((m) => ({ id: m.id, name: m.name })),
  );
  const [usingMock, setUsingMock] = useState(true);

  // Available task/model combos discovered from the result dir
  const [availableCombos, setAvailableCombos] = useState<
    Map<string, Set<string>>
  >(new Map());

  // Selection
  const [selectedTask, setSelectedTask] = useState(MOCK_TASKS[0].id);
  const [selectedModels, setSelectedModels] = useState<string[]>([
    MOCK_MODELS[0].id,
    MOCK_MODELS[1].id,
  ]);

  // Per-model predictions for the current task
  // Map from modelId -> array of predictions
  const [predictionsByModel, setPredictionsByModel] = useState<
    Map<string, ApiPrediction[]>
  >(new Map());
  const [totalByModel, setTotalByModel] = useState<Map<string, number>>(
    new Map(),
  );
  const [predictionsLoading, setPredictionsLoading] = useState(false);

  // Navigation
  const [sampleIndex, setSampleIndex] = useState(0);

  // ── Boot: fetch real tasks/models from API ──────────────────────
  useEffect(() => {
    let cancelled = false;

    async function boot() {
      try {
        const [apiTasks, apiModels, apiResults] = await Promise.all([
          fetchTasks(),
          fetchModels(),
          fetchResults(),
        ]);
        if (cancelled) return;

        // Build available combos
        const combos = new Map<string, Set<string>>();
        apiResults.forEach((r: ApiResult) => {
          if (!combos.has(r.task_id)) combos.set(r.task_id, new Set());
          combos.get(r.task_id)!.add(r.model_id);
        });
        setAvailableCombos(combos);

        // Use API tasks, but only those that have results
        const apiTaskOptions: TaskOption[] = apiTasks.map((t: ApiTask) => ({
          id: t.task_id,
          name: t.display_name,
        }));
        // Also add tasks from results that might not be in metadata
        const knownIds = new Set(apiTaskOptions.map((t) => t.id));
        for (const tid of combos.keys()) {
          if (!knownIds.has(tid)) {
            apiTaskOptions.push({ id: tid, name: tid });
          }
        }

        if (apiTaskOptions.length > 0) {
          setTasks(apiTaskOptions);
        }

        // Use all unique models from results
        const allModelIds = new Set<string>();
        apiModels.forEach((m: string) => allModelIds.add(m));
        apiResults.forEach((r: ApiResult) => allModelIds.add(r.model_id));
        const modelOptions: ModelOption[] = Array.from(allModelIds).map(
          (id) => ({ id, name: modelDisplayName(id) }),
        );
        if (modelOptions.length > 0) {
          setModels(modelOptions);
        }

        // Select first available task and its models
        const firstTask = apiTaskOptions[0]?.id;
        if (firstTask) {
          setSelectedTask(firstTask);
          const taskModels = combos.get(firstTask);
          if (taskModels && taskModels.size > 0) {
            const modelArr = Array.from(taskModels);
            setSelectedModels(modelArr.slice(0, 2));
          }
        }

        setUsingMock(false);
      } catch {
        // API unreachable — keep mock data
        setUsingMock(true);
      }
    }

    boot();
    return () => {
      cancelled = true;
    };
  }, []);

  // ── Fetch predictions when task or models change ────────────────
  const loadPredictions = useCallback(
    async (taskId: string, modelIds: string[]) => {
      if (usingMock) return;

      setPredictionsLoading(true);
      const newPreds = new Map<string, ApiPrediction[]>();
      const newTotals = new Map<string, number>();

      await Promise.all(
        modelIds.map(async (modelId) => {
          try {
            const resp = await fetchPredictions(taskId, modelId, 0, 200);
            newPreds.set(modelId, resp.predictions);
            newTotals.set(modelId, resp.total);
          } catch {
            // Model may not have predictions for this task
          }
        }),
      );

      setPredictionsByModel(newPreds);
      setTotalByModel(newTotals);
      setPredictionsLoading(false);
    },
    [usingMock],
  );

  useEffect(() => {
    if (!usingMock && selectedModels.length > 0) {
      loadPredictions(selectedTask, selectedModels);
    }
  }, [selectedTask, selectedModels, usingMock, loadPredictions]);

  // ── Compute the current sample and predictions ──────────────────

  // Determine total sample count (max across loaded models)
  let totalSamples: number;
  let currentSample: SampleData | null = null;
  let currentPredictions: PredictionEntry[] = [];

  if (usingMock) {
    // Mock path
    totalSamples = MOCK_SAMPLES.length;
    const mockSample = MOCK_SAMPLES[sampleIndex];
    if (mockSample) {
      currentSample = {
        id: mockSample.id,
        imageUrl: mockSample.imageUrl,
        placeholderColor: mockSample.placeholderColor,
        question: mockSample.question,
        groundTruth: mockSample.groundTruth,
      };
      currentPredictions = mockSample.predictions.map((p) => ({
        modelId: p.modelId,
        modelName: p.modelName,
        prediction: p.prediction,
        score: p.score,
      }));
    }
  } else {
    // Real data path
    totalSamples = Math.max(
      0,
      ...Array.from(totalByModel.values()),
    );

    // Build sample from the first model that has data at this index
    for (const [modelId, preds] of predictionsByModel.entries()) {
      const pred = preds[sampleIndex];
      if (pred && !currentSample) {
        currentSample = {
          id: pred.question_id ?? `sample-${sampleIndex}`,
          imageUrl: null,
          placeholderColor: placeholderColorFor(
            pred.question_id ?? `${sampleIndex}`,
          ),
          question: pred.input_text ?? "",
          groundTruth: pred.answer ?? "",
        };
      }
      if (pred) {
        currentPredictions.push({
          modelId,
          modelName: modelDisplayName(modelId),
          prediction: pred.text ?? "",
          score: extractScore(pred, selectedTask),
        });
      }
    }
  }

  // ── Navigation ──────────────────────────────────────────────────

  const goToPrevious = () => {
    setSampleIndex((prev) =>
      totalSamples > 0
        ? prev > 0
          ? prev - 1
          : totalSamples - 1
        : 0,
    );
  };

  const goToNext = () => {
    setSampleIndex((prev) =>
      totalSamples > 0
        ? prev < totalSamples - 1
          ? prev + 1
          : 0
        : 0,
    );
  };

  const handleTaskChange = (taskId: string) => {
    setSelectedTask(taskId);
    setSampleIndex(0);
    setPredictionsByModel(new Map());
    setTotalByModel(new Map());

    // Auto-select available models for this task
    if (!usingMock) {
      const taskModels = availableCombos.get(taskId);
      if (taskModels && taskModels.size > 0) {
        const arr = Array.from(taskModels);
        setSelectedModels(arr.slice(0, 2));
      }
    }
  };

  const task = tasks.find((t) => t.id === selectedTask);

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
          {usingMock && (
            <p
              className="mt-1 text-xs italic"
              style={{ color: "#87867f" }}
            >
              API unavailable — showing mock data.
            </p>
          )}
        </div>

        {/* Selector bar */}
        <div className="mb-6">
          <BrowserSelector
            selectedTask={selectedTask}
            onTaskChange={handleTaskChange}
            selectedModels={selectedModels}
            onModelsChange={setSelectedModels}
            tasks={tasks}
            models={models}
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
              {totalSamples > 0
                ? `${totalSamples} samples in dataset`
                : "No samples loaded"}
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
                predictions={currentPredictions}
                selectedModels={selectedModels}
                loading={predictionsLoading}
              />
            </div>
          </>
        )}

        {/* Empty state when no sample */}
        {!currentSample && !predictionsLoading && (
          <div
            className="mb-6 flex items-center justify-center rounded-xl border py-16"
            style={{ borderColor: "#e8e6dc", backgroundColor: "#faf9f5" }}
          >
            <span className="text-sm" style={{ color: "#87867f" }}>
              Select a task and model to browse predictions.
            </span>
          </div>
        )}

        {/* Navigation */}
        {totalSamples > 0 && (
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
        )}
      </div>
    </div>
  );
}
