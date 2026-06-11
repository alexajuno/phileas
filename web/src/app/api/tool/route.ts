import { NextResponse, type NextRequest } from "next/server";

import {
  callDaemon,
  DaemonError,
  DaemonUnavailableError,
} from "@/lib/daemon";
import type { MemoryItem } from "@/lib/types";

export const dynamic = "force-dynamic";

// Allowlist — mirrors tool_runner.TOOL_NAMES on the Python side. These are the
// read-only recall-family tools; mutation tools are deliberately excluded so the
// playground can never write to the live memory store.
const TOOLS = new Set([
  "recall_recent",
  "timeline",
  "about",
  "list_day_memories",
  "serendipity",
  "hydrate",
  "thread",
  "find_entities",
  "scopes",
]);

type DaemonToolResult = {
  items: Array<Record<string, unknown>>;
  text: string;
  tokens: number;
};

type Body = {
  tool?: unknown;
  args?: unknown;
};

export async function POST(request: NextRequest) {
  let body: Body;
  try {
    body = (await request.json()) as Body;
  } catch {
    return NextResponse.json({ error: "invalid json" }, { status: 400 });
  }

  const tool = typeof body.tool === "string" ? body.tool : "";
  if (!TOOLS.has(tool)) {
    return NextResponse.json(
      { error: `unknown tool: ${tool || "(none)"}` },
      { status: 400 },
    );
  }

  const args =
    body.args && typeof body.args === "object" && !Array.isArray(body.args)
      ? (body.args as Record<string, unknown>)
      : {};

  const t0 = performance.now();
  let result: DaemonToolResult;
  try {
    result = await callDaemon<DaemonToolResult>(tool, args);
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

  // Enrich memory-bearing items into full cards (mirrors /api/recall). Tools that
  // return non-memory rows (find_entities) or no ids resolve to an empty card
  // set, and the UI falls back to the verbatim model-output string. Re-map by id
  // so the tool's own ordering is preserved (memories_by_ids sorts by date).
  const rawItems = Array.isArray(result.items) ? result.items : [];
  const ids = rawItems
    .map((it) => (typeof it?.id === "string" ? it.id : null))
    .filter((id): id is string => id !== null);
  const rows =
    ids.length > 0
      ? await callDaemon<MemoryItem[]>("memories_by_ids", { ids })
      : [];
  const byId = new Map(rows.map((r) => [r.id, r]));
  const cards: MemoryItem[] = [];
  const seen = new Set<string>();
  for (const id of ids) {
    if (seen.has(id)) continue;
    seen.add(id);
    const row = byId.get(id);
    if (row) cards.push(row);
  }

  return NextResponse.json(
    {
      text: result.text ?? "",
      cards,
      count: rawItems.length,
      tokens: result.tokens ?? 0,
      elapsed_ms,
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
