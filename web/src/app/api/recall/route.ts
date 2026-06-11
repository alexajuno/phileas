import { NextResponse, type NextRequest } from "next/server";

import {
  callDaemon,
  DaemonError,
  DaemonUnavailableError,
} from "@/lib/daemon";
import type { MemoryItem } from "@/lib/types";

export const dynamic = "force-dynamic";

const MAX_QUERY_LEN = 500;
const TOP_K_MIN = 1;
const TOP_K_MAX = 100;
const TOP_K_DEFAULT = 30;

type RecallHit = {
  id: string;
  summary: string;
  type: string;
  importance: number;
  score: number;
  created_at: string | null;
};

export type RecallResult = MemoryItem & { score: number };

type Body = {
  query?: unknown;
  top_k?: unknown;
  memory_type?: unknown;
  min_importance?: unknown;
};

function clampTopK(v: unknown): number {
  const n = typeof v === "number" ? v : Number.parseInt(String(v ?? ""), 10);
  if (!Number.isFinite(n)) return TOP_K_DEFAULT;
  return Math.min(TOP_K_MAX, Math.max(TOP_K_MIN, Math.trunc(n)));
}

export async function POST(request: NextRequest) {
  let body: Body;
  try {
    body = (await request.json()) as Body;
  } catch {
    return NextResponse.json({ error: "invalid json" }, { status: 400 });
  }

  const query = typeof body.query === "string" ? body.query.trim() : "";
  if (!query) {
    return NextResponse.json({ error: "query required" }, { status: 400 });
  }
  if (query.length > MAX_QUERY_LEN) {
    return NextResponse.json(
      { error: `query too long (max ${MAX_QUERY_LEN})` },
      { status: 400 },
    );
  }

  const top_k = clampTopK(body.top_k);
  const memory_type =
    typeof body.memory_type === "string" && body.memory_type
      ? body.memory_type
      : undefined;
  const min_importance =
    typeof body.min_importance === "number"
      ? body.min_importance
      : Number.parseInt(String(body.min_importance ?? ""), 10);

  const params: Record<string, unknown> = { query, top_k };
  if (memory_type) params.memory_type = memory_type;
  if (Number.isFinite(min_importance) && min_importance > 0) {
    params.min_importance = min_importance;
  }

  const t0 = performance.now();
  let hits: RecallHit[];
  try {
    hits = await callDaemon<RecallHit[]>("recall", params);
  } catch (err) {
    if (err instanceof DaemonUnavailableError) {
      return NextResponse.json({ error: err.message }, { status: 503 });
    }
    if (err instanceof DaemonError) {
      return NextResponse.json({ error: err.message }, { status: 502 });
    }
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
  const elapsed_ms = Math.round(performance.now() - t0);

  const ids = hits.map((h) => h.id);
  const rows =
    ids.length > 0
      ? await callDaemon<MemoryItem[]>("memories_by_ids", { ids })
      : [];
  const byId = new Map(rows.map((r) => [r.id, r]));

  const items: RecallResult[] = [];
  for (const hit of hits) {
    const row = byId.get(hit.id);
    if (!row) continue; // memory may have been forgotten between recall and enrich
    items.push({ ...row, score: hit.score });
  }

  return NextResponse.json(
    { items, elapsed_ms },
    { headers: { "Cache-Control": "no-store" } },
  );
}
