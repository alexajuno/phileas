"use client";

import type { TraceRow } from "@/lib/types";

export type ResolvedMemory = {
  id: string;
  summary: string | null;
  memory_type: string | null;
  importance: number | null;
  created_at: string | null;
};

export type DetailPayload = {
  trace: TraceRow;
  memories: ResolvedMemory[];
};

export const GATHER_PATHS = [
  "keyword",
  "semantic",
  "graph",
  "raw_text",
] as const;

export const PATH_COLORS: Record<string, string> = {
  keyword: "bg-sky-500",
  semantic: "bg-emerald-500",
  graph: "bg-violet-500",
  raw_text: "bg-amber-500",
};

export const PATH_LABEL: Record<string, string> = {
  keyword: "Keyword",
  semantic: "Semantic",
  graph: "Graph",
  raw_text: "Raw text",
};

export function formatLatency(ms: number | null): string {
  if (ms == null) return "—";
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatChars(n: number | null): string {
  if (n == null) return "—";
  if (n < 1000) return `${n}`;
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}

export function gatherSources(row: TraceRow): Record<string, number> {
  const ex = row.extra;
  if (!ex || typeof ex !== "object") return {};
  const gs = (ex as Record<string, unknown>).gather_sources;
  if (!gs || typeof gs !== "object") return {};
  return gs as Record<string, number>;
}

export function hopHistogram(row: TraceRow): Record<string, number> {
  const ex = row.extra;
  if (!ex || typeof ex !== "object") return {};
  const hd = (ex as Record<string, unknown>).hop_distribution;
  if (!hd || typeof hd !== "object") return {};
  return hd as Record<string, number>;
}

export function FractionBar({
  fractions,
}: {
  fractions: Record<string, number>;
}) {
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
            title={`${path}: ${pct.toFixed(0)}%`}
          />
        );
      })}
    </div>
  );
}

export function HopBars({
  fractions,
}: {
  fractions: Record<string, number>;
}) {
  const keys = Object.keys(fractions).sort((a, b) => Number(a) - Number(b));
  if (!keys.length) {
    return <p className="text-xs text-muted-foreground">no hop data</p>;
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
