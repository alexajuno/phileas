import { execFileSync } from "node:child_process";
import Link from "next/link";

import { IngestionView } from "@/components/ingestion-view";
import { MonitoringView } from "@/components/monitoring-view";
import { SiteHeader } from "@/components/site-header";
import { todayLocal } from "@/lib/day";
import { cn } from "@/lib/utils";
import { callDaemon, DaemonUnavailableError } from "@/lib/daemon";
import type {
  AggregateResult,
  CompareResult,
  IngestionEventListItem,
  IngestionHealth,
  TraceRow,
} from "@/lib/types";

export const dynamic = "force-dynamic";

type Tab = "recall" | "ingestion";

type SearchParams = Promise<{
  date?: string | string[];
  source?: string | string[];
  cutoff?: string | string[];
  tab?: string | string[];
}>;

function firstString(v: string | string[] | undefined): string | undefined {
  return Array.isArray(v) ? v[0] : v;
}

function parseTab(v: string | undefined): Tab {
  return v === "ingestion" ? "ingestion" : "recall";
}

function defaultCutoffIso(): string {
  try {
    const out = execFileSync(
      "git",
      ["log", "-1", "--format=%ct", "main"],
      {
        cwd: process.cwd(),
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      },
    ).trim();
    const seconds = Number.parseInt(out, 10);
    if (Number.isFinite(seconds) && seconds > 0) {
      return new Date(seconds * 1000).toISOString();
    }
  } catch {
    // fall through
  }
  return new Date(Date.now() - 86400 * 1000).toISOString();
}

const SUB_TABS: { key: Tab; label: string; href: string }[] = [
  { key: "recall", label: "Recall pipeline", href: "/monitoring?tab=recall" },
  { key: "ingestion", label: "Ingestion", href: "/monitoring?tab=ingestion" },
];

function SubTabs({ active }: { active: Tab }) {
  return (
    <nav className="-mt-2 mb-5 flex items-center gap-1 text-xs">
      {SUB_TABS.map((t) => {
        const isActive = t.key === active;
        return (
          <Link
            key={t.key}
            href={t.href}
            className={cn(
              "rounded-full border px-3 py-1 transition-colors",
              isActive
                ? "border-foreground bg-foreground text-background"
                : "border-border/60 bg-card/40 text-muted-foreground hover:border-border",
            )}
          >
            {t.label}
          </Link>
        );
      })}
    </nav>
  );
}

export default async function Page({
  searchParams,
}: {
  searchParams: SearchParams;
}) {
  const sp = await searchParams;
  const tab = parseTab(firstString(sp.tab));
  const date = firstString(sp.date) ?? todayLocal();
  const source = firstString(sp.source) ?? "";
  const cutoffParam = firstString(sp.cutoff);
  const initialCutoff =
    cutoffParam && !Number.isNaN(new Date(cutoffParam).getTime())
      ? new Date(cutoffParam).toISOString()
      : defaultCutoffIso();

  let recallRows: TraceRow[] = [];
  let recallAggregate: AggregateResult | null = null;
  let recallCompare: CompareResult | null = null;
  let ingestionHealth: IngestionHealth | null = null;
  let ingestionEvents: IngestionEventListItem[] = [];
  let error: string | null = null;
  let unavailable = false;

  try {
    if (tab === "recall") {
      recallRows = await callDaemon<TraceRow[]>("metrics_traces", {
        date,
        source: source || undefined,
        limit: 200,
      });
      recallAggregate = await callDaemon<AggregateResult>("metrics_aggregate", {
        days: 7,
      });
      recallCompare = await callDaemon<CompareResult>("metrics_compare", {
        cutoff: initialCutoff,
        source: "engine.recall_raw",
        window_days: 7,
      });
    } else {
      ingestionHealth = await callDaemon<IngestionHealth>("ingestion_health");
      ingestionEvents = await callDaemon<IngestionEventListItem[]>(
        "ingestion_events",
        { limit: 50 },
      );
    }
  } catch (err) {
    if (err instanceof DaemonUnavailableError) {
      unavailable = true;
    } else {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  return (
    <div className="mx-auto w-full max-w-6xl px-5 pb-16 sm:px-6">
      <SiteHeader currentTab="monitoring" />
      <SubTabs active={tab} />

      {error ? (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <p className="font-medium">Could not load monitoring data</p>
          <p className="mt-1 font-mono text-xs opacity-80">{error}</p>
        </div>
      ) : tab === "ingestion" ? (
        <IngestionView
          initialHealth={ingestionHealth}
          initialEvents={ingestionEvents}
          unavailable={unavailable}
        />
      ) : (
        <MonitoringView
          initialDate={date}
          initialSource={source}
          initialRows={recallRows}
          aggregate={recallAggregate}
          initialCompare={recallCompare}
          initialCutoff={initialCutoff}
          unavailable={unavailable}
        />
      )}
    </div>
  );
}
