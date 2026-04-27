import { MonitoringView } from "@/components/monitoring-view";
import { SiteHeader } from "@/components/site-header";
import { todayLocal } from "@/lib/day";
import { aggregateRecent, listTraces, type TraceRow } from "@/lib/metrics-db";

export const dynamic = "force-dynamic";

type SearchParams = Promise<{
  date?: string | string[];
  source?: string | string[];
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
  const date = firstString(sp.date) ?? todayLocal();
  const source = firstString(sp.source) ?? "";

  let initialRows: TraceRow[] = [];
  let aggregate: Awaited<ReturnType<typeof aggregateRecent>> | null = null;
  let error: string | null = null;
  let unavailable = false;
  try {
    initialRows = listTraces({ date, source: source || undefined, limit: 200 });
    aggregate = aggregateRecent(7);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes("unable to open") || msg.includes("does not exist")) {
      unavailable = true;
    } else {
      error = msg;
    }
  }

  return (
    <div className="mx-auto w-full max-w-6xl px-5 pb-16 sm:px-6">
      <SiteHeader currentTab="monitoring" />

      {error ? (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <p className="font-medium">Could not load monitoring data</p>
          <p className="mt-1 font-mono text-xs opacity-80">{error}</p>
        </div>
      ) : (
        <MonitoringView
          initialDate={date}
          initialSource={source}
          initialRows={initialRows}
          aggregate={aggregate}
          unavailable={unavailable}
        />
      )}
    </div>
  );
}
