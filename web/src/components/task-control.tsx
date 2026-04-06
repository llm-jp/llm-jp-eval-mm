"use client";

import { useState } from "react";
import { MODELS, TASKS } from "@/lib/mock-runs";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { Play, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

type Backend = "transformers" | "vllm";

export function TaskControl() {
  const [selectedModel, setSelectedModel] = useState(MODELS[0].id);
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(
    new Set(TASKS.slice(0, 4).map((t) => t.id))
  );
  const [backend, setBackend] = useState<Backend>("vllm");
  const [dropdownOpen, setDropdownOpen] = useState(false);

  function toggleTask(id: string) {
    setSelectedTasks((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <div className="rounded-lg border border-[#362d59] bg-[#150f23] p-4">
      <h2 className="text-sm font-semibold text-white mb-4">Task Control</h2>

      {/* Model selector */}
      <div className="mb-4">
        <label className="block text-xs text-[#e5e7eb] mb-1.5">Model</label>
        <div className="relative">
          <button
            type="button"
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex w-full items-center justify-between rounded-md border border-[#362d59] bg-[#1f1633] px-3 py-2 text-sm text-white hover:border-[#6a5fc1] transition-colors"
          >
            <span className="truncate">
              {MODELS.find((m) => m.id === selectedModel)?.label}
            </span>
            <ChevronDown
              className={cn(
                "size-4 text-[#e5e7eb] transition-transform",
                dropdownOpen && "rotate-180"
              )}
            />
          </button>
          {dropdownOpen && (
            <div className="absolute z-10 mt-1 w-full rounded-md border border-[#362d59] bg-[#1f1633] py-1 shadow-lg">
              {MODELS.map((model) => (
                <button
                  key={model.id}
                  type="button"
                  onClick={() => {
                    setSelectedModel(model.id);
                    setDropdownOpen(false);
                  }}
                  className={cn(
                    "block w-full px-3 py-1.5 text-left text-sm transition-colors",
                    model.id === selectedModel
                      ? "bg-[#6a5fc1]/20 text-white"
                      : "text-[#e5e7eb] hover:bg-[#362d59]/50"
                  )}
                >
                  {model.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Task checkboxes */}
      <div className="mb-4">
        <label className="block text-xs text-[#e5e7eb] mb-1.5">Tasks</label>
        <div className="grid gap-2">
          {TASKS.map((task) => (
            <label
              key={task.id}
              className="flex items-center gap-2 cursor-pointer group"
            >
              <Checkbox
                checked={selectedTasks.has(task.id)}
                onCheckedChange={() => toggleTask(task.id)}
                className="border-[#362d59] data-checked:border-[#c2ef4e] data-checked:bg-[#c2ef4e] data-checked:text-[#150f23]"
              />
              <span className="text-sm text-[#e5e7eb] group-hover:text-white transition-colors">
                {task.label}
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Backend toggle */}
      <div className="mb-5">
        <label className="block text-xs text-[#e5e7eb] mb-1.5">Backend</label>
        <div className="flex rounded-md border border-[#362d59] overflow-hidden">
          {(["transformers", "vllm"] as const).map((b) => (
            <button
              key={b}
              type="button"
              onClick={() => setBackend(b)}
              className={cn(
                "flex-1 py-1.5 text-xs font-medium transition-colors",
                backend === b
                  ? "bg-[#6a5fc1] text-white"
                  : "bg-[#1f1633] text-[#e5e7eb] hover:bg-[#362d59]/50"
              )}
            >
              {b === "vllm" ? "vLLM" : "Transformers"}
            </button>
          ))}
        </div>
      </div>

      {/* Start button */}
      <Button
        className="w-full bg-[#c2ef4e] text-[#150f23] font-semibold hover:bg-[#c2ef4e]/80 border-none"
      >
        <Play className="size-4" />
        Start Evaluation
      </Button>
    </div>
  );
}
