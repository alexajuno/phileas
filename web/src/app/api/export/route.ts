import { NextResponse, type NextRequest } from "next/server";

import { isValidDay, localDayOf } from "@/lib/day";
import { fetchMemoriesForExport } from "@/lib/queries";
import type { MemoryItem } from "@/lib/types";

export const dynamic = "force-dynamic";

type ExportRow = {
  id: string;
  name: string;
  body: string | null;
  memory_type: string;
  importance: number;
  created_at: string;
};

function toExportRow(m: MemoryItem): ExportRow {
  return {
    id: m.id,
    name: m.summary,
    body: m.raw_text,
    memory_type: m.memory_type,
    importance: m.importance,
    created_at: m.created_at,
  };
}

function toMarkdown(items: MemoryItem[]): string {
  // Group by local day, newest day first; within a day newest memory first (already sorted DESC).
  const byDay = new Map<string, MemoryItem[]>();
  for (const m of items) {
    const day = localDayOf(m.created_at);
    const bucket = byDay.get(day) ?? [];
    bucket.push(m);
    byDay.set(day, bucket);
  }
  const days = [...byDay.keys()].sort((a, b) => (a < b ? 1 : a > b ? -1 : 0));

  const lines: string[] = ["# Phileas memory export", ""];
  for (const day of days) {
    lines.push(`## ${day}`, "");
    for (const m of byDay.get(day)!) {
      const summary = m.summary.replace(/\s+/g, " ").trim();
      lines.push(`- **${summary}** _(${m.memory_type} · importance ${m.importance})_`);
      if (m.raw_text) {
        const body = m.raw_text.trim().replace(/\n/g, "\n  ");
        lines.push(`  ${body}`);
      }
    }
    lines.push("");
  }
  return lines.join("\n");
}

export async function GET(request: NextRequest) {
  const sp = request.nextUrl.searchParams;
  const format = (sp.get("format") ?? "json").toLowerCase();
  if (format !== "json" && format !== "markdown") {
    return NextResponse.json(
      { error: "format must be 'json' or 'markdown'" },
      { status: 400 },
    );
  }

  const from = sp.get("from") ?? undefined;
  const to = sp.get("to") ?? undefined;
  for (const [k, v] of [["from", from], ["to", to]] as const) {
    if (v && !isValidDay(v)) {
      return NextResponse.json({ error: `invalid ${k} date` }, { status: 400 });
    }
  }

  const type = sp.get("type") ?? undefined;
  const minRaw = sp.get("min");
  const minImportance = minRaw ? Number.parseInt(minRaw, 10) : undefined;
  if (minImportance !== undefined && (!Number.isFinite(minImportance) || minImportance < 1 || minImportance > 10)) {
    return NextResponse.json({ error: "min must be between 1 and 10" }, { status: 400 });
  }

  let items: MemoryItem[];
  try {
    items = fetchMemoriesForExport({ from, to, type, minImportance });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const status = message.includes("unable to open") ? 503 : 500;
    return NextResponse.json({ error: message }, { status });
  }

  const stamp = new Date().toISOString().slice(0, 10);
  const rangeTag = from || to ? `_${from ?? "start"}_${to ?? stamp}` : "";
  const typeTag = type ? `_${type}` : "";
  const baseName = `phileas-memories${rangeTag}${typeTag}`;

  if (format === "markdown") {
    const body = toMarkdown(items);
    return new Response(body, {
      headers: {
        "Content-Type": "text/markdown; charset=utf-8",
        "Content-Disposition": `attachment; filename="${baseName}.md"`,
        "Cache-Control": "no-store",
      },
    });
  }

  const body = JSON.stringify(items.map(toExportRow), null, 2);
  return new Response(body, {
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Content-Disposition": `attachment; filename="${baseName}.json"`,
      "Cache-Control": "no-store",
    },
  });
}
