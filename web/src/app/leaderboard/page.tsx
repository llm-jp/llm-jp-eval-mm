import { LEADERBOARD_DATA, TASKS, LAST_UPDATED } from "@/lib/mock-data";
import { StatsSummary } from "@/components/stats-summary";
import { LeaderboardTable } from "@/components/leaderboard-table";
import { StaticLeaderboard } from "@/components/landing/static-leaderboard";
import { IS_STATIC_EXPORT } from "@/lib/base-path";

export default function LeaderboardPage() {
  if (IS_STATIC_EXPORT) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 lg:px-8">
        <StaticLeaderboard />
      </div>
    );
  }
  return (
    <div className="mx-auto max-w-[1600px] px-4 py-10 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-[#061b31]">
          Leaderboard
        </h1>
        <p className="mt-2 text-sm text-[#64748d]">
          Compare model performance across English and Japanese multimodal
          benchmarks.
        </p>
      </div>
      <div className="mb-8">
        <StatsSummary
          modelCount={LEADERBOARD_DATA.length}
          taskCount={TASKS.length}
          lastUpdated={LAST_UPDATED}
        />
      </div>
      <LeaderboardTable data={LEADERBOARD_DATA} />
    </div>
  );
}
