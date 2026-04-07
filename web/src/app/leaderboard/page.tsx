import { LEADERBOARD_DATA, TASKS, LAST_UPDATED } from "@/lib/mock-data";
import { StatsSummary } from "@/components/stats-summary";
import { LeaderboardTable } from "@/components/leaderboard-table";

export default function LeaderboardPage() {
  return (
    <div className="mx-auto max-w-[1600px] px-4 py-10 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-[#061b31]">
          Leaderboard
        </h1>
        <p className="mt-2 text-sm text-[#64748d]">
          Compare model performance across English and Japanese multimodal
          benchmarks.
        </p>
      </div>

      {/* Stats cards */}
      <div className="mb-8">
        <StatsSummary
          modelCount={LEADERBOARD_DATA.length}
          taskCount={TASKS.length}
          lastUpdated={LAST_UPDATED}
        />
      </div>

      {/* Leaderboard table */}
      <LeaderboardTable data={LEADERBOARD_DATA} />
    </div>
  );
}
