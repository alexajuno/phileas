import "server-only";

import { getDb } from "./phileas-db";
import { localDayBoundsAsUtcIso } from "./day";
import type { DayCount, MemoryItem } from "./types";

type Row = Omit<MemoryItem, "tags"> & { tags: string };

function parseTags(raw: string | null | undefined): string[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}

export function fetchMemoriesForDay(day: string): MemoryItem[] {
  const { startIso, endIso } = localDayBoundsAsUtcIso(day);
  const rows = getDb()
    .prepare<[string, string], Row>(
      `SELECT id, summary, memory_type, importance, tier, status,
              access_count, reinforcement_count, last_reinforced,
              raw_text, tags, daily_ref, source_session_id,
              created_at, updated_at
         FROM memory_items
        WHERE status = 'active'
          AND created_at >= ?
          AND created_at <  ?
        ORDER BY created_at DESC`
    )
    .all(startIso, endIso);
  return rows.map((r) => ({ ...r, tags: parseTags(r.tags) }));
}

export function searchMemories(rawQuery: string, limit = 100): MemoryItem[] {
  const terms = rawQuery
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 8);
  if (terms.length === 0) return [];

  const clauses: string[] = [];
  const params: string[] = [];
  for (const t of terms) {
    clauses.push(
      "(summary LIKE ? ESCAPE '\\' OR raw_text LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\')",
    );
    const like = `%${t.replace(/([\\%_])/g, "\\$1")}%`;
    params.push(like, like, like);
  }

  const sql = `SELECT id, summary, memory_type, importance, tier, status,
                      access_count, reinforcement_count, last_reinforced,
                      raw_text, tags, daily_ref, source_session_id,
                      created_at, updated_at
                 FROM memory_items
                WHERE status = 'active'
                  AND ${clauses.join(" AND ")}
                ORDER BY created_at DESC
                LIMIT ?`;

  const rows = getDb()
    .prepare<(string | number)[], Row>(sql)
    .all(...params, limit);
  return rows.map((r) => ({ ...r, tags: parseTags(r.tags) }));
}

export function fetchDaysWithCounts(limit = 60): DayCount[] {
  // SQLite substring is UTC-stored; collapse into local-day buckets server-side
  // by pulling all created_at values then grouping. Cheap for <100k rows.
  const rows = getDb()
    .prepare<[], { created_at: string }>(
      `SELECT created_at FROM memory_items WHERE status = 'active'`
    )
    .all();
  const buckets = new Map<string, number>();
  for (const r of rows) {
    const d = new Date(r.created_at);
    const p = (n: number) => String(n).padStart(2, "0");
    const key = `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
    buckets.set(key, (buckets.get(key) ?? 0) + 1);
  }
  return Array.from(buckets.entries())
    .map(([day, count]) => ({ day, count }))
    .sort((a, b) => (a.day < b.day ? 1 : a.day > b.day ? -1 : 0))
    .slice(0, limit);
}
