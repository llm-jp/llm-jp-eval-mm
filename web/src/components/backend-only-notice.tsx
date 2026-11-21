import Link from "next/link";

export function BackendOnlyNotice({ page }: { page: string }) {
  return (
    <div className="mx-auto max-w-2xl px-4 py-24 text-center sm:px-6 lg:px-8">
      <h1 className="text-2xl font-semibold tracking-tight">
        {page} is not available on the public site
      </h1>
      <p className="mt-4 text-sm text-muted-foreground">
        This view talks to the FastAPI backend and is only available when
        running the dev dashboard locally. Start the backend and run{" "}
        <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
          pnpm dev
        </code>{" "}
        in <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">web/</code>.
      </p>
      <div className="mt-8">
        <Link href="/" className="text-sm underline-offset-4 hover:underline">
          ← Back to overview
        </Link>
      </div>
    </div>
  );
}
