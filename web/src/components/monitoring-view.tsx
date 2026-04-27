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
import type { AggregateResult, TraceRow } from "@/lib/metrics-db";

const SOURCES = [
  { value: "", label: "All sources" },
  { value: "hook_dispatch", label: "Hook dispatch" },
  { value: "engine.recall_raw", label: "engine.recall_raw" },
  { value: "engine.recall_recent", label: "engine.recall_recent" },
  { value: "engine.recall", label: "engine.recall" },
];

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

export function MonitoringView({
  initialDate,
  initialSource,
  initialRows,
  aggregate,
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
                  <th className="px-3 py-2 text-right">Pool</th>
                  <th className="px-3 py-2 text-right">Latency</th>
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 ? (
                  <tr>
                    <td
                      colSpan={6}
                      className="px-3 py-8 text-center text-sm text-muted-foreground"
                    >
                      No traces for this filter.
                    </td>
                  </tr>
                ) : (
                  rows.map((row) => (
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
                      <td className="max-w-[28rem] truncate px-3 py-2 text-foreground">
                        {row.query ?? <span className="text-muted-foreground">—</span>}
                      </td>
                      <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-xs">
                        {row.candidate_count ?? "—"}
                      </td>
                      <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-xs text-muted-foreground">
                        {formatChars(row.pool_chars)}
                      </td>
                      <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-xs">
                        {formatLatency(row.latency_ms)}
                      </td>
                    </tr>
                  ))
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
          <pre className="mt-1 whitespace-pre-wrap break-words rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-xs">
            {trace.query}
          </pre>
        </div>
      )}

      {trace.extra && Object.keys(trace.extra).length > 0 && (
        <div>
          <Label>Extra</Label>
          <pre className="mt-1 max-h-48 overflow-auto rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-xs">
            {JSON.stringify(trace.extra, null, 2)}
          </pre>
        </div>
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
