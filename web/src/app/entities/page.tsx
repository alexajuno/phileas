import Link from "next/link";

import { EntityListView } from "@/components/entity-list-view";
import { ThemeToggle } from "@/components/theme-toggle";
import { DaemonUnavailableError } from "@/lib/daemon";
import { listEntities } from "@/lib/graph";
import type { EntitySummary } from "@/lib/types";

export const dynamic = "force-dynamic";

type SearchParams = Promise<{
  q?: string | string[];
  type?: string | string[];
}>;

function firstString(v: string | string[] | undefined): string | undefined {
  return Array.isArray(v) ? v[0] : v;
}

export default async function Page({
  searchParams,
}: {
  searchParams: SearchParams;
}) {
  const sp = await searchParams;
  const q = (firstString(sp.q) ?? "").trim();
  const type = firstString(sp.type)?.trim() || null;

  let initialItems: EntitySummary[] = [];
  let unavailable = false;
  let error: string | null = null;
  try {
    const all = await listEntities({ limit: 500, type_filter: type });
    initialItems = q
      ? all.filter(
          (e) =>
            e.name.toLowerCase().includes(q.toLowerCase()) ||
            e.aliases.some((a) => a.toLowerCase().includes(q.toLowerCase())),
        )
      : all;
  } catch (err) {
    if (err instanceof DaemonUnavailableError) {
      unavailable = true;
    } else {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-5 pb-16 pt-6 sm:px-6">
      <header className="mb-6 flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-base font-medium tracking-tight">
            Phileas <span className="text-muted-foreground">· entities</span>
          </h1>
          <p className="mt-1 text-xs text-muted-foreground">
            Knowledge graph nodes — people, places, concepts mentioned across memories.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <ThemeToggle />
          <Link
            href="/search"
            className="text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            ↗ search
          </Link>
          <Link
            href="/"
            className="text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            ← today
          </Link>
        </div>
      </header>

      {error ? (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <p className="font-medium">Could not load entities</p>
          <p className="mt-1 font-mono text-xs opacity-80">{error}</p>
        </div>
      ) : (
        <EntityListView
          initialQuery={q}
          initialType={type}
          initialItems={initialItems}
          unavailable={unavailable}
        />
      )}
    </div>
  );
}
