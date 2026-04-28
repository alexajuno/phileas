"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import type {
  AggregateResult,
  BucketStats,
  CompareResult,
  TraceRow,
} from "@/lib/metrics-db";

const SOURCES = [
  { value: "", label: "All sources" },
  { value: "hook_dispatch", label: "Hook dispatch" },
  { value: "engine.recall_raw", label: "engine.recall_raw" },
  { value: "engine.recall_recent", label: "engine.recall_recent" },
  { value: "engine.recall", label: "engine.recall" },
];

const GATHER_PATHS = ["keyword", "semantic", "graph", "raw_text"] as const;

const PATH_COLORS: Record<string, string> = {
  keyword: "bg-sky-500",
  semantic: "bg-emerald-500",
  graph: "bg-violet-500",
  raw_text: "bg-amber-500",
};

const PATH_LABEL: Record<string, string> = {
  keyword: "Keyword",
  semantic: "Semantic",
  graph: "Graph",
  raw_text: "Raw text",
};

type ResolvedMemory = {
  id: string;
  summary: string | null;
  memory_type: string | null;
  importance: number | null;
  created_at: string | null;
};

type DetailPayload = {
  trace: TraceRow;
  memories: ResolvedMemory[];
};

type Props = {
  initialDate: string;
  initialSource: string;
  initialRows: TraceRow[];
  aggregate: AggregateResult | null;
  initialCompare: CompareResult | null;
  initialCutoff: string;
  unavailable?: boolean;
};

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function formatLatency(ms: number | null): string {
  if (ms == null) return "—";
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function formatChars(n: number | null): string {
  if (n == null) return "—";
  if (n < 1000) return `${n}`;
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}

function estTokens(chars: number | null): number | null {
  if (chars == null) return null;
  return Math.round(chars / 4);
}

function sourceBadgeClass(source: string): string {
  if (source === "hook_dispatch")
    return "bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30";
  if (source === "engine.recall_raw")
    return "bg-sky-500/15 text-sky-300 ring-1 ring-sky-500/30";
  if (source === "engine.recall_recent")
    return "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30";
  if (source === "engine.recall")
    return "bg-violet-500/15 text-violet-300 ring-1 ring-violet-500/30";
  return "bg-muted text-muted-foreground";
}

function gatherSources(row: TraceRow): Record<string, number> {
  const ex = row.extra;
  if (!ex || typeof ex !== "object") return {};
  const gs = (ex as Record<string, unknown>).gather_sources;
  if (!gs || typeof gs !== "object") return {};
  return gs as Record<string, number>;
}

function hopHistogram(row: TraceRow): Record<string, number> {
  const ex = row.extra;
  if (!ex || typeof ex !== "object") return {};
  const hd = (ex as Record<string, unknown>).hop_distribution;
  if (!hd || typeof hd !== "object") return {};
  return hd as Record<string, number>;
}

function topHop(hist: Record<string, number>): string {
  let bestKey: string | null = null;
  let bestVal = -1;
  for (const [k, v] of Object.entries(hist)) {
    if (k === "None" || k == null) continue;
    if (v > bestVal) {
      bestVal = v;
      bestKey = k;
    }
  }
  if (bestKey == null) return "—";
  return `h${bestKey}`;
}

function SourceMixBar({
  hist,
  className,
}: {
  hist: Record<string, number>;
  className?: string;
}) {
  const total = Object.values(hist).reduce((a, c) => a + c, 0);
  if (total <= 0) {
    return <div className={cn("h-2 w-24 rounded bg-muted/40", className)} />;
  }
  return (
    <div
      className={cn(
        "flex h-2 w-24 overflow-hidden rounded bg-muted/40",
        className,
      )}
      title={Object.entries(hist)
        .map(([k, v]) => `${k}: ${v}`)
        .join(" · ")}
    >
      {GATHER_PATHS.map((path) => {
        const v = hist[path] ?? 0;
        if (v <= 0) return null;
        const pct = (v / total) * 100;
        return (
          <div
            key={path}
            className={PATH_COLORS[path] ?? "bg-muted-foreground"}
            style={{ width: `${pct}%` }}
          />
        );
      })}
    </div>
  );
}

function FractionBar({ fractions }: { fractions: Record<string, number> }) {
  const total = Object.values(fractions).reduce((a, c) => a + c, 0);
  if (total <= 0) {
    return <div className="h-3 w-full rounded bg-muted/40" />;
  }
  return (
    <div className="flex h-3 w-full overflow-hidden rounded bg-muted/40">
      {GATHER_PATHS.map((path) => {
        const v = fractions[path] ?? 0;
        if (v <= 0) return null;
        const pct = (v / total) * 100;
        return (
          <div
            key={path}
            className={PATH_COLORS[path] ?? "bg-muted-foreground"}
            style={{ width: `${pct}%` }}
            title={`${path}: ${(pct).toFixed(0)}%`}
          />
        );
      })}
    </div>
  );
}

function HopBars({ fractions }: { fractions: Record<string, number> }) {
  const keys = Object.keys(fractions).sort((a, b) => Number(a) - Number(b));
  if (!keys.length) {
    return (
      <p className="text-xs text-muted-foreground">no hop data</p>
    );
  }
  const max = Math.max(...keys.map((k) => fractions[k]));
  return (
    <div className="space-y-1">
      {keys.map((k) => {
        const v = fractions[k];
        const pct = max > 0 ? (v / max) * 100 : 0;
        return (
          <div key={k} className="flex items-center gap-2 text-[11px]">
            <span className="w-6 font-mono text-muted-foreground">h{k}</span>
            <div className="flex-1">
              <div
                className="h-2 rounded bg-violet-500"
                style={{ width: `${pct}%`, minWidth: pct > 0 ? "2px" : 0 }}
              />
            </div>
            <span className="w-12 text-right font-mono tabular-nums text-muted-foreground">
              {(v * 100).toFixed(0)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

function ChangePill({
  before,
  after,
  unit,
  invert = true,
}: {
  before: number | null;
  after: number | null;
  unit: string;
  invert?: boolean;
}) {
  if (before == null || after == null) return null;
  const delta = after - before;
  if (before === 0) return null;
  const pct = (delta / before) * 100;
  const isImprovement = invert ? delta < 0 : delta > 0;
  const cls = Math.abs(pct) < 0.5
    ? "text-muted-foreground"
    : isImprovement
      ? "text-emerald-400"
      : "text-rose-400";
  const sign = delta > 0 ? "+" : "";
  return (
    <span className={cn("font-mono text-[11px]", cls)}>
      {sign}
      {Math.abs(pct) >= 100
        ? `${pct.toFixed(0)}%`
        : `${pct.toFixed(1)}%`}{" "}
      <span className="text-muted-foreground">({sign}
        {Math.round(delta)}{unit})</span>
    </span>
  );
}

function StatPair({
  label,
  before,
  after,
  format,
  unit,
}: {
  label: string;
  before: number | null;
  after: number | null;
  format: (v: number | null) => string;
  unit: string;
}) {
  return (
    <div className="grid grid-cols-[8rem_1fr_1fr_auto] items-center gap-2 rounded-md border border-border/40 bg-muted/10 px-2 py-1.5">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className="font-mono text-sm tabular-nums">{format(before)}</div>
      <div className="font-mono text-sm tabular-nums">{format(after)}</div>
      <ChangePill before={before} after={after} unit={unit} invert />
    </div>
  );
}

function localCutoff(iso: string): string {
  const d = new Date(iso);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}T${hh}:${mi}`;
}

export function MonitoringView({
  initialDate,
  initialSource,
  initialRows,
  aggregate,
  initialCompare,
  initialCutoff,
  unavailable = false,
}: Props) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [date, setDate] = useState(initialDate);
  const [source, setSource] = useState(initialSource);
  const [rows, setRows] = useState<TraceRow[]>(initialRows);
  const [loading, setLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<DetailPayload | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const detailReqId = useRef(0);

  const [compareOpen, setCompareOpen] = useState(false);
  const [cutoff, setCutoff] = useState(initialCutoff);
  const [compare, setCompare] = useState<CompareResult | null>(initialCompare);
  const [compareLoading, setCompareLoading] = useState(false);

  const openTrace = useCallback(async (id: number) => {
    setSelectedId(id);
    setDetail(null);
    setDetailLoading(true);
    const reqId = ++detailReqId.current;
    try {
      const res = await fetch(`/api/monitoring/traces/${id}`, { cache: "no-store" });
      if (!res.ok) throw new Error(await res.text());
      const data: DetailPayload = await res.json();
      if (detailReqId.current === reqId) setDetail(data);
    } catch (err) {
      console.error("trace detail failed", err);
      if (detailReqId.current === reqId) setDetail(null);
    } finally {
      if (detailReqId.current === reqId) setDetailLoading(false);
    }
  }, []);

  const refetch = useCallback(
    async (nextDate: string, nextSource: string) => {
      setLoading(true);
      try {
        const params = new URLSearchParams({ date: nextDate, limit: "200" });
        if (nextSource) params.set("source", nextSource);
        const res = await fetch(`/api/monitoring/traces?${params}`, {
          cache: "no-store",
        });
        if (!res.ok) throw new Error(await res.text());
        const data: TraceRow[] = await res.json();
        setRows(data);
      } catch (err) {
        console.error("monitoring refetch failed", err);
        setRows([]);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const refetchCompare = useCallback(async (nextCutoff: string) => {
    setCompareLoading(true);
    try {
      const params = new URLSearchParams({
        cutoff: nextCutoff,
        source: "engine.recall_raw",
        days: "7",
      });
      const res = await fetch(`/api/monitoring/compare?${params}`, {
        cache: "no-store",
      });
      if (!res.ok) throw new Error(await res.text());
      const data: CompareResult = await res.json();
      setCompare(data);
    } catch (err) {
      console.error("compare fetch failed", err);
    } finally {
      setCompareLoading(false);
    }
  }, []);

  const updateUrl = useCallback(
    (nextDate: string, nextSource: string) => {
      const next = new URLSearchParams(searchParams.toString());
      next.set("date", nextDate);
      if (nextSource) next.set("source", nextSource);
      else next.delete("source");
      router.replace(`/monitoring?${next.toString()}`, { scroll: false });
    },
    [router, searchParams],
  );

  const onDateChange = (next: string) => {
    setDate(next);
    updateUrl(next, source);
    void refetch(next, source);
  };
  const onSourceChange = (next: string) => {
    setSource(next);
    updateUrl(date, next);
    void refetch(date, next);
  };
  const onCutoffChange = (next: string) => {
    const iso = new Date(next).toISOString();
    setCutoff(iso);
    void refetchCompare(iso);
  };

  const aggCards = useMemo(() => {
    if (!aggregate) return null;
    const all = aggregate.by_source;
    const total = aggregate.total;
    const lat = all.flatMap((s) => (s.p50 != null ? [s] : []));
    const overallP50 = lat.length
      ? lat.reduce((a, c) => a + (c.p50 ?? 0), 0) / lat.length
      : null;
    const overallP90 = lat.length
      ? lat.reduce((a, c) => a + (c.p90 ?? 0), 0) / lat.length
      : null;
    const avgPool = lat.length
      ? lat.reduce((a, c) => a + (c.avg_pool_chars ?? 0), 0) / lat.length
      : null;
    return { total, overallP50, overallP90, avgPool };
  }, [aggregate]);

  return (
    <div>
      {unavailable ? (
        <div className="rounded-lg border border-border/60 bg-card/40 px-4 py-3 text-sm text-muted-foreground">
          <p className="font-medium text-foreground">No metrics database yet</p>
          <p className="mt-1">
            <code>~/.phileas/metrics.db</code> doesn&apos;t exist or is empty.
            Run a memory-flavored prompt with the daemon up to populate it.
          </p>
        </div>
      ) : (
        <>
          <section className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Card label="Traces (7d)" value={aggCards?.total ?? "—"} />
            <Card
              label="Avg p50 latency"
              value={
                aggCards?.overallP50 != null ? formatLatency(aggCards.overallP50) : "—"
              }
            />
            <Card
              label="Avg p90 latency"
              value={
                aggCards?.overallP90 != null ? formatLatency(aggCards.overallP90) : "—"
              }
            />
            <Card
              label="Avg pool size"
              value={
                aggCards?.avgPool != null
                  ? `${formatChars(Math.round(aggCards.avgPool))} chars`
                  : "—"
              }
            />
          </section>

          <ComparePanel
            open={compareOpen}
            onToggle={() => setCompareOpen((v) => !v)}
            cutoff={cutoff}
            onCutoffChange={onCutoffChange}
            compare={compare}
            loading={compareLoading}
          />

          <section className="mb-4 flex flex-wrap items-center gap-3 rounded-lg border border-border/60 bg-card/30 px-3 py-2">
            <label className="text-xs text-muted-foreground" htmlFor="m-date">
              Date
            </label>
            <input
              id="m-date"
              type="date"
              value={date}
              onChange={(e) => onDateChange(e.target.value)}
              className="rounded-md border border-border/60 bg-background px-2 py-1 text-sm"
            />
            <label className="text-xs text-muted-foreground" htmlFor="m-source">
              Source
            </label>
            <select
              id="m-source"
              value={source}
              onChange={(e) => onSourceChange(e.target.value)}
              className="rounded-md border border-border/60 bg-background px-2 py-1 text-sm"
            >
              {SOURCES.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
            <span className="ml-auto text-xs text-muted-foreground">
              {loading ? "loading…" : `${rows.length} rows`}
            </span>
          </section>

          <section className="overflow-hidden rounded-lg border border-border/60 bg-card/30">
            <table className="w-full text-sm">
              <thead className="bg-muted/40 text-xs uppercase tracking-wide text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 text-left">Time</th>
                  <th className="px-3 py-2 text-left">Source</th>
                  <th className="px-3 py-2 text-left">Query</th>
                  <th className="px-3 py-2 text-right">Cand</th>
                  <th className="px-3 py-2 text-right">Ret</th>
                  <th className="px-3 py-2 text-left">Mix</th>
                  <th className="px-3 py-2 text-right">Hop</th>
                  <th className="px-3 py-2 text-right">Latency</th>
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 ? (
                  <tr>
                    <td
                      colSpan={8}
                      className="px-3 py-8 text-center text-sm text-muted-foreground"
                    >
                      No traces for this filter.
                    </td>
                  </tr>
                ) : (
                  rows.map((row) => {
                    const hist = gatherSources(row);
                    const hops = hopHistogram(row);
                    const returned = row.returned_ids?.length ?? 0;
                    return (
                      <tr
                        key={row.id}
                        onClick={() => void openTrace(row.id)}
                        className="cursor-pointer border-t border-border/40 transition-colors hover:bg-muted/30"
                      >
                        <td className="whitespace-nowrap px-3 py-2 font-mono text-xs text-muted-foreground">
                          {formatTime(row.created_at)}
                        </td>
                        <td className="px-3 py-2">
                          <span
                            className={cn(
                              "inline-block rounded px-2 py-0.5 text-[11px]",
                              sourceBadgeClass(row.source),
                            )}
                          >
                            {row.source}
                          </span>
                        </td>
                        <td className="max-w-[22rem] truncate px-3 py-2 text-foreground">
                          {row.query ?? <span className="text-muted-foreground">—</span>}
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-xs">
                          {row.candidate_count ?? "—"}
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-xs text-muted-foreground">
                          {returned || "—"}
                        </td>
                        <td className="px-3 py-2">
                          <SourceMixBar hist={hist} />
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-xs text-muted-foreground">
                          {topHop(hops)}
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-xs">
                          {formatLatency(row.latency_ms)}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </section>
        </>
      )}

      <Dialog
        open={selectedId != null}
        onOpenChange={(open) => {
          if (!open) {
            setSelectedId(null);
            setDetail(null);
          }
        }}
      >
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>Trace #{selectedId}</DialogTitle>
          </DialogHeader>
          {detailLoading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : detail ? (
            <TraceDetail detail={detail} />
          ) : (
            <p className="text-sm text-muted-foreground">No data.</p>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

function ComparePanel({
  open,
  onToggle,
  cutoff,
  onCutoffChange,
  compare,
  loading,
}: {
  open: boolean;
  onToggle: () => void;
  cutoff: string;
  onCutoffChange: (next: string) => void;
  compare: CompareResult | null;
  loading: boolean;
}) {
  return (
    <section className="mb-4 rounded-lg border border-border/60 bg-card/30">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">
            Before / after — engine.recall_raw
          </span>
          {compare ? (
            <span className="font-mono text-[11px] text-muted-foreground">
              n={compare.before.count} → {compare.after.count}
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-3">
          {loading ? (
            <span className="text-xs text-muted-foreground">loading…</span>
          ) : null}
          <span className="text-xs text-muted-foreground">
            {open ? "hide" : "show"}
          </span>
        </div>
      </button>
      {open ? (
        <div className="border-t border-border/60 px-3 py-3">
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <label
              className="text-xs text-muted-foreground"
              htmlFor="m-cutoff"
            >
              Cutoff (default: most recent commit on main)
            </label>
            <input
              id="m-cutoff"
              type="datetime-local"
              value={localCutoff(cutoff)}
              onChange={(e) => onCutoffChange(e.target.value)}
              className="rounded-md border border-border/60 bg-background px-2 py-1 text-sm"
            />
            <span className="font-mono text-[11px] text-muted-foreground">
              {new Date(cutoff).toLocaleString()}
            </span>
          </div>

          {compare == null ? (
            <p className="text-sm text-muted-foreground">No compare data.</p>
          ) : (
            <CompareBody compare={compare} />
          )}
        </div>
      ) : null}
    </section>
  );
}

function CompareBody({ compare }: { compare: CompareResult }) {
  const { before, after } = compare;
  const empty = before.count === 0 && after.count === 0;
  if (empty) {
    return (
      <p className="text-sm text-muted-foreground">
        No <code>engine.recall_raw</code> traces in the last 7 days.
      </p>
    );
  }
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-[8rem_1fr_1fr_auto] gap-2 px-2 text-[11px] uppercase tracking-wide text-muted-foreground">
        <div />
        <div>Before</div>
        <div>After</div>
        <div className="text-right">Δ</div>
      </div>

      <div className="space-y-1.5">
        <StatPair
          label="Sample size"
          before={before.count}
          after={after.count}
          format={(v) => (v == null ? "—" : `${v}`)}
          unit=""
        />
        <StatPair
          label="Cand p50"
          before={before.candidate_p50}
          after={after.candidate_p50}
          format={(v) => (v == null ? "—" : Math.round(v).toString())}
          unit=""
        />
        <StatPair
          label="Cand p95"
          before={before.candidate_p95}
          after={after.candidate_p95}
          format={(v) => (v == null ? "—" : Math.round(v).toString())}
          unit=""
        />
        <StatPair
          label="Latency p50"
          before={before.latency_p50}
          after={after.latency_p50}
          format={formatLatency}
          unit=" ms"
        />
        <StatPair
          label="Latency p95"
          before={before.latency_p95}
          after={after.latency_p95}
          format={formatLatency}
          unit=" ms"
        />
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <BucketBlock title="Before" stats={before} />
        <BucketBlock title="After" stats={after} />
      </div>

      <Legend />
    </div>
  );
}

function BucketBlock({ title, stats }: { title: string; stats: BucketStats }) {
  return (
    <div className="rounded-md border border-border/40 bg-muted/10 p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {title}
        </h3>
        <span className="font-mono text-[11px] text-muted-foreground">
          n={stats.count}
        </span>
      </div>
      <div className="mb-3">
        <div className="mb-1 text-[11px] uppercase tracking-wide text-muted-foreground">
          Mean source mix
        </div>
        <FractionBar fractions={stats.source_mix} />
      </div>
      <div>
        <div className="mb-1 text-[11px] uppercase tracking-wide text-muted-foreground">
          Mean hop distribution
        </div>
        <HopBars fractions={stats.hop_distribution} />
      </div>
    </div>
  );
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
      {GATHER_PATHS.map((path) => (
        <span key={path} className="inline-flex items-center gap-1">
          <span className={cn("inline-block h-2 w-3 rounded", PATH_COLORS[path])} />
          {PATH_LABEL[path]}
        </span>
      ))}
    </div>
  );
}

function Card({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-border/60 bg-card/30 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className="mt-0.5 text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function TraceDetail({ detail }: { detail: DetailPayload }) {
  const { trace, memories } = detail;
  const tokens = estTokens(trace.pool_chars);
  const hist = gatherSources(trace);
  const hops = hopHistogram(trace);
  const histTotal = Object.values(hist).reduce((a, c) => a + c, 0);
  const hopTotal = Object.entries(hops)
    .filter(([k]) => k !== "None")
    .reduce((a, [, v]) => a + v, 0);
  return (
    <div className="space-y-4 text-sm">
      <div className="grid grid-cols-2 gap-3">
        <KV label="Source" value={trace.source} />
        <KV label="When" value={new Date(trace.created_at).toLocaleString()} />
        <KV label="Latency" value={formatLatency(trace.latency_ms)} />
        <KV
          label="Candidates"
          value={trace.candidate_count?.toString() ?? "—"}
        />
        <KV
          label="Pool size"
          value={
            trace.pool_chars != null
              ? `${formatChars(trace.pool_chars)} chars · ~${tokens} tok`
              : "—"
          }
        />
        <KV label="Returned" value={(trace.returned_ids?.length ?? 0).toString()} />
      </div>

      {trace.query && (
        <div>
          <Label>Query</Label>
          <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-xs">
            {trace.query}
          </pre>
        </div>
      )}

      {histTotal > 0 && (
        <div>
          <Label>Source mix ({histTotal} matches)</Label>
          <div className="mt-1.5 space-y-1.5">
            <FractionBar fractions={hist} />
            <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
              {GATHER_PATHS.map((path) => {
                const v = hist[path] ?? 0;
                if (v <= 0) return null;
                const pct = (v / histTotal) * 100;
                return (
                  <span key={path} className="inline-flex items-center gap-1">
                    <span
                      className={cn("inline-block h-2 w-3 rounded", PATH_COLORS[path])}
                    />
                    {PATH_LABEL[path]}: {v} ({pct.toFixed(0)}%)
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {hopTotal > 0 && (
        <div>
          <Label>Hop distribution (graph distance)</Label>
          <div className="mt-1.5">
            <HopBars
              fractions={Object.fromEntries(
                Object.entries(hops)
                  .filter(([k]) => k !== "None")
                  .map(([k, v]) => [k, v / hopTotal]),
              )}
            />
          </div>
        </div>
      )}

      {trace.extra && Object.keys(trace.extra).length > 0 && (
        <details className="rounded-md border border-border/40 bg-muted/10">
          <summary className="cursor-pointer px-2 py-1 text-[11px] uppercase tracking-wide text-muted-foreground">
            Raw extra
          </summary>
          <pre className="max-h-48 overflow-auto p-2 font-mono text-xs">
            {JSON.stringify(trace.extra, null, 2)}
          </pre>
        </details>
      )}

      {memories.length > 0 && (
        <div>
          <Label>Memories returned ({memories.length})</Label>
          <ul className="mt-1 max-h-72 space-y-1 overflow-auto rounded-md border border-border/60 bg-muted/20 p-2 text-xs">
            {memories.map((m) => (
              <li key={m.id} className="font-mono">
                <span className="text-muted-foreground">[{m.id.slice(0, 8)}]</span>{" "}
                {m.memory_type ? (
                  <span className="text-amber-400">[{m.memory_type}]</span>
                ) : (
                  <span className="text-destructive">[deleted?]</span>
                )}{" "}
                {m.summary ?? <span className="text-muted-foreground">(missing)</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border/40 bg-muted/20 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className="mt-0.5 font-mono text-xs">{value}</div>
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
      {children}
    </div>
  );
}
