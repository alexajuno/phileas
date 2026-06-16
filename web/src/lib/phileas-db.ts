import "server-only";

import { homedir } from "node:os";
import { join } from "node:path";
import Database, { type Database as DB } from "better-sqlite3";

let cached: DB | null = null;
let cachedPath: string | null = null;

function resolveDbPath(): string {
  const home = process.env.PHILEAS_HOME ?? join(homedir(), ".phileas");
  return join(home, "memory.db");
}

export function getDb(): DB {
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

export function dbPath(): string {
  return resolveDbPath();
}

// ---- Ingestion monitoring ----

export type IngestionEventRow = {
  id: string;
  text: string;
  received_at: string;
};

export type IngestionEventListItem = Omit<IngestionEventRow, "text"> & {
  text_preview: string;
};

export type IngestionHealth = {
  events_received_1h: number;
  events_received_24h: number;
  events_total: number;
};

export type LinkedMemoryRow = {
  id: string;
  summary: string;
  memory_type: string;
  created_at: string;
};

const PREVIEW_CHARS = 240;

function preview(text: string): string {
  if (text.length <= PREVIEW_CHARS) return text;
  return text.slice(0, PREVIEW_CHARS) + "…";
}

export function fetchIngestionHealth(): IngestionHealth {
  const db = getDb();
  const oneHourAgo = new Date(Date.now() - 3600 * 1000).toISOString();
  const oneDayAgo = new Date(Date.now() - 86400 * 1000).toISOString();

  const recv1h = db
    .prepare<[string], { count: number }>(
      `SELECT COUNT(*) AS count FROM events WHERE received_at >= ?`,
    )
    .get(oneHourAgo);

  const recv24h = db
    .prepare<[string], { count: number }>(
      `SELECT COUNT(*) AS count FROM events WHERE received_at >= ?`,
    )
    .get(oneDayAgo);

  const total = db
    .prepare<[], { count: number }>(`SELECT COUNT(*) AS count FROM events`)
    .get();

  return {
    events_received_1h: recv1h?.count ?? 0,
    events_received_24h: recv24h?.count ?? 0,
    events_total: total?.count ?? 0,
  };
}

export type ListIngestionOpts = {
  limit?: number;
};

export function listIngestionEvents(
  opts: ListIngestionOpts = {},
): IngestionEventListItem[] {
  const limit = Math.min(Math.max(opts.limit ?? 50, 1), 500);
  const rows = getDb()
    .prepare<[number], IngestionEventRow>(
      `SELECT id, text, received_at
         FROM events
        ORDER BY received_at DESC
        LIMIT ?`,
    )
    .all(limit);

  return rows.map(({ text, ...rest }) => ({
    ...rest,
    text_preview: preview(text),
  }));
}

export function fetchIngestionEvent(id: string): {
  event: IngestionEventRow;
  memories: LinkedMemoryRow[];
} | null {
  const db = getDb();
  const event = db
    .prepare<[string], IngestionEventRow>(
      `SELECT id, text, received_at FROM events WHERE id = ?`,
    )
    .get(id);
  if (!event) return null;

  const memories = db
    .prepare<[string], LinkedMemoryRow>(
      `SELECT id, summary, memory_type, created_at
         FROM memory_items
        WHERE source_event_id = ?
          AND status = 'active'
        ORDER BY created_at ASC`,
    )
    .all(id);

  return { event, memories };
}
