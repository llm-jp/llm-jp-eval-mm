import { Card, CardContent } from "@/components/ui/card";
import { BarChart3, Layers, CalendarDays } from "lucide-react";

interface StatsSummaryProps {
  modelCount: number;
  taskCount: number;
  lastUpdated: string;
}

const stats = (props: StatsSummaryProps) => [
  {
    label: "Models",
    value: props.modelCount,
    icon: BarChart3,
    description: "Evaluated models",
  },
  {
    label: "Tasks",
    value: props.taskCount,
    icon: Layers,
    description: "Benchmark tasks",
  },
  {
    label: "Last Updated",
    value: props.lastUpdated,
    icon: CalendarDays,
    description: "Latest evaluation run",
  },
];

export function StatsSummary(props: StatsSummaryProps) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      {stats(props).map((stat) => (
        <Card
          key={stat.label}
          className="border-[#e5edf5] bg-white"
          style={{
            boxShadow:
              "0 2px 12px rgba(50,50,93,0.06), 0 1px 3px rgba(0,0,0,0.03)",
          }}
        >
          <CardContent className="flex items-center gap-4">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-[#f8f6ff]">
              <stat.icon className="size-5 text-[#533afd]" />
            </div>
            <div>
              <p className="text-2xl font-bold tabular-nums text-[#061b31]">
                {stat.value}
              </p>
              <p className="text-xs text-[#64748d]">{stat.description}</p>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
