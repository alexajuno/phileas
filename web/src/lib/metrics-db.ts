import "server-only";

import { homedir } from "node:os";
import { join } from "node:path";
import Database, { type Database as DB } from "better-sqlite3";

let cached: DB | null = null;
let cachedPath: string | null = null;

function resolveDbPath(): string {
  const home = process.env.PHILEAS_HOME ?? join(homedir(), ".phileas");
  return join(home, "metrics.db");
}

export function getMetricsDb(): DB {
  const path = resolveDbPath();
  if (cached && cachedPath === path) return cached;
  if (cached) {
    cached.close();
    cached = null;
  }
  const db = new Database(path, { readonly: true, fileMustExist: true });
  db.pragma("query_only = ON");
  cached = db;
  cachedPath = path;
  return db;
}

export function metricsDbPath(): string {
  return resolveDbPath();
}

export type TraceSource =
  | "hook_dispatch"
  | "engine.recall"
  | "engine.recall_raw"
  | "engine.recall_recent";

export interface TraceRow {
  id: number;
  created_at: string;
  source: TraceSource | string;
  query: string | null;
  latency_ms: number | null;
  candidate_count: number | null;
  returned_ids: string[] | null;
  pool_chars: number | null;
  extra: Record<string, unknown> | null;
}

interface RawTraceRow {
  id: number;
  created_at: string;
  source: string;
  query: string | null;
  latency_ms: number | null;
  candidate_count: number | null;
  returned_ids: string | null;
  pool_chars: number | null;
  extra: string | null;
}

function parseJson<T>(value: string | null): T | null {
  if (!value) return null;
  try {
    return JSON.parse(value) as T;
  } catch {
    return null;
  }
}

function hydrate(row: RawTraceRow): TraceRow {
  return {
    id: row.id,
    created_at: row.created_at,
    source: row.source,
    query: row.query,
    latency_ms: row.latency_ms,
    candidate_count: row.candidate_count,
    returned_ids: parseJson<string[]>(row.returned_ids),
    pool_chars: row.pool_chars,
    extra: parseJson<Record<string, unknown>>(row.extra),
  };
}

export interface ListTracesArgs {
  date?: string;
  limit?: number;
  source?: string;
}

export function listTraces(args: ListTracesArgs = {}): TraceRow[] {
  const db = getMetricsDb();
  const limit = Math.min(Math.max(args.limit ?? 200, 1), 1000);
  const clauses: string[] = [];
  const params: Record<string, unknown> = { limit };
  if (args.date) {
    clauses.push("substr(created_at, 1, 10) = @date");
    params.date = args.date;
  }
  if (args.source) {
    clauses.push("source = @source");
    params.source = args.source;
  }
  const where = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";
  const sql = `
    SELECT id, created_at, source, query, latency_ms, candidate_count,
           returned_ids, pool_chars, extra
    FROM recall_traces
    ${where}
    ORDER BY id DESC
    LIMIT @limit
  `;
  const rows = db.prepare(sql).all(params) as RawTraceRow[];
  return rows.map(hydrate);
}

export function getTrace(id: number): TraceRow | null {
  const db = getMetricsDb();
  const row = db
    .prepare(
      `SELECT id, created_at, source, query, latency_ms, candidate_count,
              returned_ids, pool_chars, extra
       FROM recall_traces WHERE id = ?`,
    )
    .get(id) as RawTraceRow | undefined;
  if (!row) return null;
  const hydrated = hydrate(row);
  if (
    hydrated.source === "engine.recall" &&
    hydrated.latency_ms != null &&
    (!hydrated.extra || hydrated.extra.stage_timings == null)
  ) {
    const stages = lookupStageTimings(hydrated.created_at, hydrated.latency_ms);
    if (stages) {
      hydrated.extra = { ...(hydrated.extra ?? {}), stage_timings: stages };
    }
  }
  return hydrated;
}

function lookupStageTimings(
  traceCreatedAt: string,
  latencyMs: number,
): Record<string, number> | null {
  const db = getMetricsDb();
  const t = new Date(traceCreatedAt).getTime();
  if (!Number.isFinite(t)) return null;
  const windowMs = 5_000;
  const lo = new Date(t - windowMs).toISOString();
  const hi = new Date(t + windowMs).toISOString();
  const row = db
    .prepare(
      `SELECT stage_timings_json, ABS(latency_ms - ?) AS lat_delta,
              ABS(strftime('%s', created_at) - strftime('%s', ?)) AS t_delta
       FROM recall_events
       WHERE created_at BETWEEN ? AND ?
         AND stage_timings_json IS NOT NULL
         AND ABS(latency_ms - ?) < 1
       ORDER BY lat_delta ASC, t_delta ASC
       LIMIT 1`,
    )
    .get(latencyMs, traceCreatedAt, lo, hi, latencyMs) as
    | { stage_timings_json: string | null }
    | undefined;
  if (!row || !row.stage_timings_json) return null;
  try {
    const parsed = JSON.parse(row.stage_timings_json) as Record<string, unknown>;
    const out: Record<string, number> = {};
    for (const [k, v] of Object.entries(parsed)) {
      if (typeof v === "number") out[k] = v;
    }
    return out;
  } catch {
    return null;
  }
}

export interface AggregateRow {
  source: string;
  count: number;
  p50: number | null;
  p90: number | null;
  p99: number | null;
  avg_pool_chars: number | null;
  avg_candidates: number | null;
}

export interface AggregateResult {
  since: string;
  by_source: AggregateRow[];
  total: number;
}

function percentile(sorted: number[], p: number): number | null {
  if (!sorted.length) return null;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((p / 100) * sorted.length)));
  return sorted[idx];
}

export interface BucketStats {
  count: number;
  candidate_p50: number | null;
  candidate_p95: number | null;
  latency_p50: number | null;
  latency_p95: number | null;
  source_mix: Record<string, number>;
  hop_distribution: Record<string, number>;
}

export interface CompareResult {
  cutoff: string;
  source: string;
  since: string;
  before: BucketStats;
  after: BucketStats;
}

interface CompareRawRow {
  created_at: string;
  latency_ms: number | null;
  candidate_count: number | null;
  extra: string | null;
}

function median(sorted: number[]): number | null {
  if (!sorted.length) return null;
  const mid = sorted.length >>> 1;
  if (sorted.length % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function p95(sorted: number[]): number | null {
  return percentile(sorted, 95);
}

function meanFractionMap(maps: Array<Record<string, number>>): Record<string, number> {
  if (!maps.length) return {};
  const fractions: Record<string, number[]> = {};
  for (const m of maps) {
    const total = Object.values(m).reduce((a, c) => a + c, 0);
    if (total <= 0) continue;
    for (const [key, val] of Object.entries(m)) {
      (fractions[key] ??= []).push(val / total);
    }
  }
  const out: Record<string, number> = {};
  for (const [key, values] of Object.entries(fractions)) {
    if (!values.length) continue;
    out[key] = values.reduce((a, c) => a + c, 0) / maps.length;
  }
  return out;
}

function bucketize(rows: CompareRawRow[]): BucketStats {
  const cand: number[] = [];
  const lat: number[] = [];
  const sources: Array<Record<string, number>> = [];
  const hops: Array<Record<string, number>> = [];
  for (const r of rows) {
    if (r.candidate_count != null) cand.push(r.candidate_count);
    if (r.latency_ms != null) lat.push(r.latency_ms);
    if (r.extra) {
      try {
        const ex = JSON.parse(r.extra) as Record<string, unknown>;
        const gs = ex.gather_sources;
        if (gs && typeof gs === "object") {
          sources.push(gs as Record<string, number>);
        }
        const hd = ex.hop_distribution;
        if (hd && typeof hd === "object") {
          const filtered: Record<string, number> = {};
          for (const [k, v] of Object.entries(hd as Record<string, number>)) {
            if (k === "None" || k == null) continue;
            filtered[k] = v;
          }
          hops.push(filtered);
        }
      } catch {
        // skip
      }
    }
  }
  cand.sort((a, b) => a - b);
  lat.sort((a, b) => a - b);
  return {
    count: rows.length,
    candidate_p50: median(cand),
    candidate_p95: p95(cand),
    latency_p50: median(lat),
    latency_p95: p95(lat),
    source_mix: meanFractionMap(sources),
    hop_distribution: meanFractionMap(hops),
  };
}

export interface CompareArgs {
  cutoffIso: string;
  source?: string;
  windowDays?: number;
}

export function compareTraces(args: CompareArgs): CompareResult {
  const db = getMetricsDb();
  const source = args.source ?? "engine.recall_raw";
  const windowDays = args.windowDays ?? 7;
  const since = new Date(Date.now() - windowDays * 86400 * 1000).toISOString();
  const rows = db
    .prepare(
      `SELECT created_at, latency_ms, candidate_count, extra
       FROM recall_traces
       WHERE source = ? AND created_at >= ?`,
    )
    .all(source, since) as CompareRawRow[];
  const before: CompareRawRow[] = [];
  const after: CompareRawRow[] = [];
  for (const r of rows) {
    if (r.created_at < args.cutoffIso) before.push(r);
    else after.push(r);
  }
  return {
    cutoff: args.cutoffIso,
    source,
    since,
    before: bucketize(before),
    after: bucketize(after),
  };
}

export function aggregateRecent(days = 7): AggregateResult {
  const db = getMetricsDb();
  const since = new Date(Date.now() - days * 86400 * 1000).toISOString();
  const rows = db
    .prepare(
      `SELECT source, latency_ms, candidate_count, pool_chars
       FROM recall_traces
       WHERE created_at >= ?`,
    )
    .all(since) as Array<{
      source: string;
      latency_ms: number | null;
      candidate_count: number | null;
      pool_chars: number | null;
    }>;

  const buckets = new Map<string, { lat: number[]; pool: number[]; cand: number[] }>();
  for (const r of rows) {
    const b = buckets.get(r.source) ?? { lat: [], pool: [], cand: [] };
    if (r.latency_ms != null) b.lat.push(r.latency_ms);
    if (r.pool_chars != null) b.pool.push(r.pool_chars);
    if (r.candidate_count != null) b.cand.push(r.candidate_count);
    buckets.set(r.source, b);
  }

  const by_source: AggregateRow[] = [];
  for (const [source, b] of buckets) {
    const sorted = [...b.lat].sort((x, y) => x - y);
    const avg = (xs: number[]) =>
      xs.length ? xs.reduce((a, c) => a + c, 0) / xs.length : null;
    by_source.push({
      source,
      count: b.lat.length || b.cand.length || 0,
      p50: percentile(sorted, 50),
      p90: percentile(sorted, 90),
      p99: percentile(sorted, 99),
      avg_pool_chars: avg(b.pool),
      avg_candidates: avg(b.cand),
    });
  }
  by_source.sort((a, b) => b.count - a.count);
  return { since, by_source, total: rows.length };
}
