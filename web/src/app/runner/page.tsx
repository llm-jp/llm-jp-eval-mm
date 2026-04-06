import { MOCK_GPUS } from "@/lib/mock-gpu";
import { GpuCard } from "@/components/gpu-card";
import { TaskControl } from "@/components/task-control";
import { ResultsMatrix } from "@/components/results-matrix";
import { RunProgress } from "@/components/run-progress";

export default function RunnerPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Page header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold tracking-tight text-white">
          Runner Dashboard
        </h1>
        <p className="mt-1 text-sm text-[#e5e7eb]">
          Monitor GPUs, configure tasks, and track evaluation progress.
        </p>
      </div>

      {/* GPU cards row */}
      <section className="mb-6">
        <div className="grid gap-3 grid-cols-2 lg:grid-cols-4">
          {MOCK_GPUS.map((gpu) => (
            <GpuCard key={gpu.id} gpu={gpu} />
          ))}
        </div>
      </section>

      {/* Middle row: Task Control + Results Matrix */}
      <section className="mb-6 grid gap-4 lg:grid-cols-[minmax(280px,1fr)_2fr]">
        <TaskControl />
        <ResultsMatrix />
      </section>

      {/* Progress panel */}
      <section>
        <RunProgress />
      </section>
    </div>
  );
}
