import Link from "next/link";
import { Search } from "lucide-react";

import { DaemonStatus } from "./daemon-status";
import { ThemeToggle } from "./theme-toggle";
import { cn } from "@/lib/utils";

export type HeaderTab = "today" | "entities";

const TABS: { key: HeaderTab; label: string; href: string }[] = [
  { key: "today", label: "Memories", href: "/" },
  { key: "entities", label: "Entities", href: "/entities" },
];

type Props = {
  currentTab?: HeaderTab;
};

export function SiteHeader({ currentTab }: Props) {
  return (
    <header className="mb-6 border-b border-border/60">
      <div className="flex items-center gap-4 py-3">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-sm font-semibold tracking-tight text-foreground"
        >
          Phileas
          <DaemonStatus />
        </Link>
        <div className="ml-auto flex items-center gap-2">
          <Link
            href="/search"
            aria-label="Search memories"
            title="Search memories"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-border/60 bg-card/60 text-muted-foreground transition-colors hover:border-border hover:text-foreground"
          >
            <Search className="h-3.5 w-3.5" aria-hidden />
          </Link>
          <ThemeToggle />
        </div>
      </div>

      <nav className="-mb-px flex items-stretch">
        {TABS.map((t) => {
          const active = t.key === currentTab;
          return (
            <Link
              key={t.key}
              href={t.href}
              className={cn(
                "relative px-3 py-2 text-sm transition-colors",
                active
                  ? "font-medium text-foreground after:absolute after:inset-x-3 after:bottom-0 after:h-[2px] after:rounded after:bg-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {t.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
