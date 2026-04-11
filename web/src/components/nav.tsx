"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", label: "Home" },
  { href: "/runner", label: "Runner" },
  { href: "/leaderboard", label: "Leaderboard" },
  { href: "/browser", label: "Browser" },
] as const;

export function Nav() {
  const pathname = usePathname();
  const isRunnerPage = pathname.startsWith("/runner");

  return (
    <header
      className={cn(
        "sticky top-0 z-50 border-b backdrop-blur",
        isRunnerPage
          ? "border-runner-border bg-runner-surface/95 supports-[backdrop-filter]:bg-runner-surface/80"
          : "bg-background/95 supports-[backdrop-filter]:bg-background/60",
      )}
    >
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-4 sm:px-6 lg:px-8">
        <Link
          href="/"
          className={cn(
            "text-lg font-semibold tracking-tight",
            isRunnerPage && "text-runner-text",
          )}
        >
          eval_mm
        </Link>

        <nav className="flex items-center gap-1">
          {NAV_ITEMS.map(({ href, label }) => {
            const isActive =
              href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  isRunnerPage
                    ? isActive
                      ? "bg-runner-primary/30 text-runner-text"
                      : "text-runner-text-secondary/70 hover:bg-runner-border/50 hover:text-runner-text"
                    : isActive
                      ? "bg-muted text-foreground"
                      : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
                )}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
