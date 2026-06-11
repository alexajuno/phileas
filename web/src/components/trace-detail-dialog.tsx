"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import type { TraceRow } from "@/lib/types";

import {
  type DetailPayload,
  FractionBar,
  GATHER_PATHS,
  HopBars,
  PATH_COLORS,
  PATH_LABEL,
  formatChars,
  formatLatency,
  gatherSources,
  hopHistogram,
} from "./monitoring-shared";

type PathCountKey = "path3" | "path3b" | "path4";

const PATH_COUNT_KEYS: readonly PathCountKey[] = ["path3", "path3b", "path4"];

const PATH_COUNT_LABEL: Record<PathCountKey, string> = {
  path3: "Path 3 (entity match)",
  path3b: "Path 3b (memory pivot)",
  path4: "Path 4 (entity bridge)",
};

const STAGE_ORDER = [
  "keyword",
  "semantic",
  "graph_path3",
  "graph_path3b_pivot",
  "graph_path3c_referent",
  "graph_path4_bridge",
  "raw_text",
  "events",
  "filter",
  "cosine_full",
  "rerank",
  "score_blend",
  "mmr",
  "final_score",
  "bump_access",
] as const;

const STAGE_GROUP: Record<string, "direct" | "graph" | "rank" | "post"> = {
  keyword: "direct",
  semantic: "direct",
  raw_text: "direct",
  events: "direct",
  graph_path3: "graph",
  graph_path3b_pivot: "graph",
  graph_path3c_referent: "graph",
  graph_path4_bridge: "graph",
  filter: "rank",
  cosine_full: "rank",
  rerank: "rank",
  score_blend: "rank",
  mmr: "rank",
  final_score: "post",
  bump_access: "post",
};

const STAGE_GROUP_COLOR: Record<string, string> = {
  direct: "bg-emerald-500",
  graph: "bg-violet-500",
  rank: "bg-amber-500",
  post: "bg-slate-500",
};

const STAGE_GROUP_LABEL: Record<string, string> = {
  direct: "Direct paths",
  graph: "Graph paths",
  rank: "Rank/filter",
  post: "Post",
};

function estTokens(chars: number | null): number | null {
  if (chars == null) return null;
  return Math.round(chars / 4);
}

function stageTimings(row: TraceRow): Record<string, number> {
  const ex = row.extra;
  if (!ex || typeof ex !== "object") return {};
  const st = (ex as Record<string, unknown>).stage_timings;
  if (!st || typeof st !== "object") return {};
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(st as Record<string, unknown>)) {
    if (typeof v === "number" && v > 0) out[k] = v;
  }
  return out;
}

function pathCounts(row: TraceRow): Record<PathCountKey, number> {
  const ex = row.extra;
  if (!ex || typeof ex !== "object") {
    return { path3: 0, path3b: 0, path4: 0 };
  }
  const e = ex as Record<string, unknown>;
  const read = (k: string): number => {
    const v = e[k];
    return typeof v === "number" ? v : 0;
  };
  return {
    path3: read("path3_count"),
    path3b: read("path3b_count"),
    path4: read("path4_count"),
  };
}

function uniquePathContribution(row: TraceRow): Record<string, number> {
  const ex = row.extra;
  if (!ex || typeof ex !== "object") return {};
  const u = (ex as Record<string, unknown>).result_unique_path_counts;
  if (!u || typeof u !== "object") return {};
  return u as Record<string, number>;
}

function gatherHistogram(row: TraceRow): Record<string, number> {
  const ex = row.extra;
  if (!ex || typeof ex !== "object") return {};
  const h = (ex as Record<string, unknown>).result_gather_histogram;
  if (!h || typeof h !== "object") return {};
  return h as Record<string, number>;
}

export function TraceDetailDialog({
  traceId,
  detail,
  loading,
  onClose,
}: {
  traceId: number | null;
  detail: DetailPayload | null;
  loading: boolean;
  onClose: () => void;
}) {
  return (
    <Dialog
      open={traceId != null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="flex max-h-[85vh] flex-col overflow-hidden sm:max-w-5xl">
        <DialogHeader>
          <DialogTitle>Trace #{traceId}</DialogTitle>
        </DialogHeader>
        <div className="flex-1 overflow-y-auto pr-1">
          {loading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : detail ? (
            <TraceDetail detail={detail} />
          ) : (
            <p className="text-sm text-muted-foreground">No data.</p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function TraceDetail({ detail }: { detail: DetailPayload }) {
  const { trace, memories } = detail;
  const tokens = estTokens(trace.pool_chars);
  const hist = gatherSources(trace);
  const hops = hopHistogram(trace);
  const stages = stageTimings(trace);
  const counts = pathCounts(trace);
  const unique = uniquePathContribution(trace);
  const gHist = gatherHistogram(trace);
  const hasStages = Object.keys(stages).length > 0;
  const hasPathData =
    counts.path3 > 0 || counts.path3b > 0 || counts.path4 > 0;
  const histTotal = Object.values(hist).reduce((a, c) => a + c, 0);
  const hopTotal = Object.entries(hops)
    .filter(([k]) => k !== "None")
    .reduce((a, [, v]) => a + v, 0);
  const rightHasContent = hasPathData || histTotal > 0 || hopTotal > 0;
  return (
    <div className="space-y-4 text-sm">
      <div className="grid grid-cols-3 gap-3">
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
        <KV
          label="Returned"
          value={(trace.returned_ids?.length ?? 0).toString()}
        />
      </div>

      {trace.query && (
        <div>
          <Label>Query</Label>
          <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-xs">
            {trace.query}
          </pre>
        </div>
      )}

      <div
        className={cn(
          "grid gap-4",
          hasStages && rightHasContent ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1",
        )}
      >
        {hasStages && (
          <div>
            <Label>Stage timings</Label>
            <div className="mt-1.5">
              <StageBars timings={stages} />
            </div>
          </div>
        )}
        {rightHasContent && (
          <div className="space-y-4">
            {hasPathData && (
              <div>
                <Label>Graph sub-paths</Label>
                <div className="mt-1.5">
                  <PathAttribution
                    counts={counts}
                    unique={unique}
                    gatherHist={gHist}
                  />
                </div>
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
                        <span
                          key={path}
                          className="inline-flex items-center gap-1"
                        >
                          <span
                            className={cn(
                              "inline-block h-2 w-3 rounded",
                              PATH_COLORS[path],
                            )}
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
          </div>
        )}
      </div>

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
                {m.summary ?? (
                  <span className="text-muted-foreground">(missing)</span>
                )}
              </li>
            ))}
          </ul>
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
    </div>
  );
}

function StageBars({ timings }: { timings: Record<string, number> }) {
  const entries = STAGE_ORDER.filter((k) => (timings[k] ?? 0) > 0).map((k) => ({
    name: k,
    ms: timings[k],
    group: STAGE_GROUP[k] ?? "post",
  }));
  if (!entries.length) {
    return <p className="text-xs text-muted-foreground">no stage data</p>;
  }
  const total = entries.reduce((a, e) => a + e.ms, 0);
  const max = Math.max(...entries.map((e) => e.ms));
  const groupsPresent = Array.from(new Set(entries.map((e) => e.group)));
  return (
    <div className="space-y-2">
      <div className="flex h-2 w-full overflow-hidden rounded bg-muted/40">
        {entries.map((e) => {
          const pct = (e.ms / total) * 100;
          return (
            <div
              key={e.name}
              className={STAGE_GROUP_COLOR[e.group]}
              style={{ width: `${pct}%` }}
              title={`${e.name}: ${formatLatency(e.ms)} (${pct.toFixed(1)}%)`}
            />
          );
        })}
      </div>
      <div className="space-y-0.5">
        {entries.map((e) => {
          const pct = max > 0 ? (e.ms / max) * 100 : 0;
          const shareOfTotal = (e.ms / total) * 100;
          const slow = e.ms >= 1000;
          return (
            <div
              key={e.name}
              className="grid grid-cols-[10rem_1fr_4.5rem_3rem] items-center gap-2 text-[11px]"
            >
              <span className="font-mono text-muted-foreground">{e.name}</span>
              <div className="h-2">
                <div
                  className={cn(
                    "h-2 rounded",
                    STAGE_GROUP_COLOR[e.group],
                    slow ? "opacity-100" : "opacity-60",
                  )}
                  style={{ width: `${pct}%`, minWidth: pct > 0 ? "2px" : 0 }}
                />
              </div>
              <span
                className={cn(
                  "text-right font-mono tabular-nums",
                  slow ? "text-rose-400" : "text-foreground",
                )}
              >
                {formatLatency(e.ms)}
              </span>
              <span className="text-right font-mono tabular-nums text-muted-foreground">
                {shareOfTotal.toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-3 pt-1 text-[10px] text-muted-foreground">
        {groupsPresent.map((g) => (
          <span key={g} className="inline-flex items-center gap-1">
            <span
              className={cn("inline-block h-2 w-3 rounded", STAGE_GROUP_COLOR[g])}
            />
            {STAGE_GROUP_LABEL[g]}
          </span>
        ))}
        <span className="ml-auto font-mono">total {formatLatency(total)}</span>
      </div>
    </div>
  );
}

function PathAttribution({
  counts,
  unique,
  gatherHist,
}: {
  counts: Record<PathCountKey, number>;
  unique: Record<string, number>;
  gatherHist: Record<string, number>;
}) {
  const maxCount = Math.max(1, ...PATH_COUNT_KEYS.map((k) => counts[k]));
  const totalCount = PATH_COUNT_KEYS.reduce((a, k) => a + counts[k], 0);
  if (totalCount === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        no graph sub-path candidates (entity-less query)
      </p>
    );
  }
  return (
    <div className="space-y-2">
      {PATH_COUNT_KEYS.map((k) => {
        const count = counts[k];
        const u = unique[k] ?? 0;
        const inFinal = gatherHist[k] ?? 0;
        const pct = (count / maxCount) * 100;
        const saturated = count >= 100 && u === 0;
        return (
          <div
            key={k}
            className={cn(
              "rounded-md border border-border/40 bg-muted/10 px-2 py-1.5",
              saturated && "border-rose-500/40 bg-rose-500/5",
            )}
          >
            <div className="mb-1 flex items-baseline justify-between gap-2 text-[11px]">
              <span className="font-medium text-foreground">
                {PATH_COUNT_LABEL[k]}
              </span>
              {saturated ? (
                <span className="font-mono text-rose-400">
                  ⚠ saturated · 0 unique
                </span>
              ) : null}
            </div>
            <div className="grid grid-cols-[1fr_auto_auto_auto] items-center gap-3 text-[11px]">
              <div className="h-2">
                <div
                  className="h-2 rounded bg-violet-500"
                  style={{ width: `${pct}%`, minWidth: pct > 0 ? "2px" : 0 }}
                />
              </div>
              <span className="font-mono tabular-nums text-muted-foreground">
                <span className="text-foreground">{count}</span>
                <span className="text-[10px]"> candidates</span>
              </span>
              <span className="font-mono tabular-nums text-muted-foreground">
                <span className="text-foreground">{inFinal}</span>
                <span className="text-[10px]"> in top-K</span>
              </span>
              <span className="font-mono tabular-nums text-muted-foreground">
                <span className="text-foreground">{u}</span>
                <span className="text-[10px]"> unique</span>
              </span>
            </div>
          </div>
        );
      })}
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
