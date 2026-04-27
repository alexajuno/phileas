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
  return row ? hydrate(row) : null;
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
