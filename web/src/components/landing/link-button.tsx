import { ReactNode } from "react";

export function LinkButton({ url, children }: { url: string; children: ReactNode }) {
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer noopener"
      className="inline-flex items-center gap-1.5 rounded-md border bg-card px-4 py-2 text-sm font-medium transition-colors hover:bg-accent"
    >
      {children}
    </a>
  );
}
