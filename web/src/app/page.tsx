import Link from "next/link";
import { Search } from "lucide-react";

import { MemoryList } from "@/components/memory-list";
import { ThemeToggle } from "@/components/theme-toggle";
import { isValidDay, todayLocal } from "@/lib/day";
import { fetchMemoriesForDay } from "@/lib/queries";

export const dynamic = "force-dynamic";

type SearchParams = Promise<{
  day?: string | string[];
  type?: string | string[];
  min?: string | string[];
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
  const today = todayLocal();

  const requestedDay = firstString(sp.day);
  const day =
    requestedDay && isValidDay(requestedDay) && requestedDay <= today
      ? requestedDay
      : today;

  const requestedType = firstString(sp.type);
  const initialType =
    requestedType && requestedType.length > 0 ? requestedType : null;

  const requestedMin = firstString(sp.min);
  const parsedMin = requestedMin ? Number.parseInt(requestedMin, 10) : NaN;
  const initialMin =
    Number.isFinite(parsedMin) && parsedMin >= 1 && parsedMin <= 10
      ? parsedMin
      : 1;

  let items: Awaited<ReturnType<typeof fetchMemoriesForDay>> = [];
  let error: string | null = null;
  try {
    items = fetchMemoriesForDay(day);
  } catch (err) {
    error = err instanceof Error ? err.message : String(err);
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-5 pb-16 pt-6 sm:px-6">
      <header className="mb-6 flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-base font-medium tracking-tight">
            Phileas{" "}
            <span className="text-muted-foreground">· daily memories</span>
          </h1>
          <p className="mt-1 text-xs text-muted-foreground">
            Long-term memory, captured throughout the day.
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <ThemeToggle />
          <Link
            href="/search"
            className="inline-flex items-center gap-1.5 rounded-md border border-border/60 bg-card/60 px-1.5 py-1.5 text-xs text-muted-foreground transition-colors hover:border-border hover:text-foreground"
            title="Search all memories"
          >
            <Search className="h-3.5 w-3.5" aria-hidden />
          </Link>
        </div>
      </header>

      {error ? (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <p className="font-medium">Could not read memory.db</p>
          <p className="mt-1 font-mono text-xs opacity-80">{error}</p>
          <p className="mt-2 text-xs text-muted-foreground">
            Expected at <code>~/.phileas/memory.db</code>. Set{" "}
            <code>PHILEAS_HOME</code> to override.
          </p>
        </div>
      ) : (
        <MemoryList
          initialDay={day}
          initialItems={items}
          initialType={initialType}
          initialMin={initialMin}
        />
      )}
    </div>
  );
}
