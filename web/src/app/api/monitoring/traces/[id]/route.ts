import { NextResponse } from "next/server";

import { callDaemon, daemonErrorStatus } from "@/lib/daemon";
import type { TraceRow } from "@/lib/types";

export const dynamic = "force-dynamic";

interface ResolvedMemory {
  id: string;
  summary: string | null;
  memory_type: string | null;
  importance: number | null;
  created_at: string | null;
}

async function resolveMemories(ids: string[]): Promise<ResolvedMemory[]> {
  if (!ids.length) return [];
  let rows: ResolvedMemory[] = [];
  try {
    rows = await callDaemon<ResolvedMemory[]>("memories_brief", { ids });
  } catch {
    // A brief-resolve failure shouldn't blank the trace — fall back to placeholders.
    rows = [];
  }
  const byId = new Map(rows.map((r) => [r.id, r]));
  return ids.map(
    (id) =>
      byId.get(id) ?? {
        id,
        summary: null,
        memory_type: null,
        importance: null,
        created_at: null,
      },
  );
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
    const trace = await callDaemon<TraceRow | null>("metrics_trace", { id });
    if (!trace) {
      return NextResponse.json({ error: "not found" }, { status: 404 });
    }
    const memories = await resolveMemories(trace.returned_ids ?? []);
    return NextResponse.json(
      { trace, memories },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: message },
      { status: daemonErrorStatus(err) },
    );
  }
}
