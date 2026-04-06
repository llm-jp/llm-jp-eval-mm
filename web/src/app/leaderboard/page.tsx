export default function LeaderboardPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
      <h1 className="text-3xl font-bold tracking-tight">Leaderboard</h1>
      <p className="mt-2 text-muted-foreground">
        Compare model performance across tasks and metrics.
      </p>

      <div className="mt-8 rounded-lg border bg-card p-6 text-card-foreground">
        <p className="text-sm text-muted-foreground">
          Leaderboard coming soon. This page will show sortable tables and
          visualisations of evaluation scores across models and benchmarks.
        </p>
      </div>
    </div>
  );
}
