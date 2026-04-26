import { MemoryList } from "@/components/memory-list";
import { SiteHeader } from "@/components/site-header";
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
    <div className="mx-auto w-full max-w-3xl px-5 pb-16 sm:px-6">
      <SiteHeader currentTab="today" />

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
