"use client";

import { useEffect, useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TASKS as MOCK_TASKS, MODELS as MOCK_MODELS } from "@/lib/mock-predictions";
import { fetchTasks, fetchModels, type ApiTask } from "@/lib/api";
import { cn } from "@/lib/utils";

interface TaskOption {
  id: string;
  name: string;
}

interface ModelOption {
  id: string;
  name: string;
}

interface BrowserSelectorProps {
  selectedTask: string;
  onTaskChange: (taskId: string) => void;
  selectedModels: string[];
  onModelsChange: (modelIds: string[]) => void;
  tasks?: TaskOption[];
  models?: ModelOption[];
}

export function BrowserSelector({
  selectedTask,
  onTaskChange,
  selectedModels,
  onModelsChange,
  tasks: externalTasks,
  models: externalModels,
}: BrowserSelectorProps) {
  const [tasks, setTasks] = useState<TaskOption[]>(
    externalTasks ?? MOCK_TASKS.map((t) => ({ id: t.id, name: t.name })),
  );
  const [models, setModels] = useState<ModelOption[]>(
    externalModels ?? MOCK_MODELS.map((m) => ({ id: m.id, name: m.name })),
  );

  // Fetch real data on mount if external data not provided
  useEffect(() => {
    if (externalTasks) return;
    let cancelled = false;
    fetchTasks()
      .then((apiTasks: ApiTask[]) => {
        if (cancelled) return;
        setTasks(
          apiTasks.map((t) => ({
            id: t.task_id,
            name: t.display_name,
          })),
        );
      })
      .catch(() => {
        // Keep mock data on failure
      });
    return () => {
      cancelled = true;
    };
  }, [externalTasks]);

  useEffect(() => {
    if (externalModels) return;
    let cancelled = false;
    fetchModels()
      .then((apiModels: string[]) => {
        if (cancelled) return;
        setModels(
          apiModels.map((id) => ({
            id,
            name: id.split("/").pop() ?? id,
          })),
        );
      })
      .catch(() => {
        // Keep mock data on failure
      });
    return () => {
      cancelled = true;
    };
  }, [externalModels]);

  const toggleModel = (modelId: string) => {
    if (selectedModels.includes(modelId)) {
      // Don't allow deselecting the last model
      if (selectedModels.length > 1) {
        onModelsChange(selectedModels.filter((id) => id !== modelId));
      }
    } else {
      onModelsChange([...selectedModels, modelId]);
    }
  };

  return (
    <div className="rounded-xl border p-4" style={{ borderColor: "#e8e6dc", backgroundColor: "#faf9f5" }}>
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:gap-6">
        {/* Task selector */}
        <div className="flex items-center gap-2">
          <label
            className="text-sm font-medium whitespace-nowrap"
            style={{ color: "#5e5d59" }}
          >
            Task
          </label>
          <Select value={selectedTask} onValueChange={(value) => { if (value) onTaskChange(value); }}>
            <SelectTrigger
              className="min-w-[180px]"
              style={{ borderColor: "#e8e6dc", backgroundColor: "#faf9f5" }}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {tasks.map((task) => (
                <SelectItem key={task.id} value={task.id}>
                  {task.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Model multi-select */}
        <div className="flex items-center gap-2">
          <label
            className="text-sm font-medium whitespace-nowrap"
            style={{ color: "#5e5d59" }}
          >
            Models
          </label>
          <div className="flex flex-wrap gap-1.5">
            {models.map((model) => {
              const isSelected = selectedModels.includes(model.id);
              return (
                <button
                  key={model.id}
                  type="button"
                  onClick={() => toggleModel(model.id)}
                  className={cn(
                    "rounded-full px-3 py-1 text-xs font-medium transition-all",
                    "border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-1"
                  )}
                  style={{
                    borderColor: isSelected ? "#c96442" : "#e8e6dc",
                    backgroundColor: isSelected ? "#c96442" : "#faf9f5",
                    color: isSelected ? "#faf9f5" : "#5e5d59",
                  }}
                >
                  {model.name}
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
