import { MemoryList } from "@/components/memory-list";
import { todayLocal } from "@/lib/day";
import { fetchMemoriesForDay } from "@/lib/queries";

export const dynamic = "force-dynamic";

export default function Page() {
  const day = todayLocal();
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
        <MemoryList initialDay={day} initialItems={items} />
      )}
    </div>
  );
}
