import Link from "next/link";

const SECTIONS = [
  {
    href: "/runner",
    title: "Runner Dashboard",
    description:
      "Launch and monitor benchmark runs. View GPU utilisation, task progress, and logs in real time.",
  },
  {
    href: "/leaderboard",
    title: "Leaderboard",
    description:
      "Compare model scores across tasks and metrics with sortable tables and interactive charts.",
  },
  {
    href: "/browser",
    title: "Prediction Browser",
    description:
      "Inspect per-sample inputs, model outputs, and scoring details side by side.",
  },
] as const;

export default function Home() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          eval_mm
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-lg text-muted-foreground">
          A unified evaluation toolkit for multimodal large language models.
        </p>
      </div>

      <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {SECTIONS.map(({ href, title, description }) => (
          <Link
            key={href}
            href={href}
            className="group rounded-xl border bg-card p-6 transition-colors hover:border-foreground/20 hover:bg-accent"
          >
            <h2 className="text-xl font-semibold tracking-tight group-hover:text-foreground">
              {title}
            </h2>
            <p className="mt-2 text-sm text-muted-foreground">{description}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
