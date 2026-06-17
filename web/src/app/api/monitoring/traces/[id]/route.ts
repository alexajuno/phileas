import { NextResponse } from "next/server";

import { getTrace } from "@/lib/metrics-db";
import { getDb } from "@/lib/phileas-db";

export const dynamic = "force-dynamic";

interface ResolvedMemory {
  id: string;
  summary: string | null;
  memory_type: string | null;
  created_at: string | null;
}

function resolveMemories(ids: string[]): ResolvedMemory[] {
  if (!ids.length) return [];
  const placeholders = ids.map(() => "?").join(",");
  const sql = `SELECT id, summary, memory_type, created_at
               FROM memory_items
               WHERE id IN (${placeholders})`;
  try {
    const rows = getDb().prepare(sql).all(...ids) as ResolvedMemory[];
    const byId = new Map(rows.map((r) => [r.id, r]));
    return ids.map(
      (id) =>
        byId.get(id) ?? {
          id,
          summary: null,
          memory_type: null,
          created_at: null,
        },
    );
  } catch {
    return ids.map((id) => ({
      id,
      summary: null,
      memory_type: null,
      created_at: null,
    }));
  }
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id: idRaw } = await params;
  const id = Number.parseInt(idRaw, 10);
  if (!Number.isFinite(id) || id <= 0) {
    return NextResponse.json({ error: "invalid id" }, { status: 400 });
  }
  try {
    const trace = getTrace(id);
    if (!trace) {
      return NextResponse.json({ error: "not found" }, { status: 404 });
    }
    const memories = resolveMemories(trace.returned_ids ?? []);
    return NextResponse.json(
      { trace, memories },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const status = message.includes("unable to open") ? 503 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
