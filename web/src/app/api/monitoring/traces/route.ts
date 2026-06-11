import { NextResponse, type NextRequest } from "next/server";

import { isValidDay } from "@/lib/day";
import { callDaemon, daemonErrorStatus } from "@/lib/daemon";
import type { TraceRow } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const date = searchParams.get("date") ?? undefined;
  const source = searchParams.get("source") ?? undefined;
  const limitRaw = searchParams.get("limit");
  const limit = limitRaw ? Number.parseInt(limitRaw, 10) : undefined;

  if (date && !isValidDay(date)) {
    return NextResponse.json({ error: "invalid date" }, { status: 400 });
  }
  if (limit != null && (!Number.isFinite(limit) || limit <= 0)) {
    return NextResponse.json({ error: "invalid limit" }, { status: 400 });
  }

  try {
    const rows = await callDaemon<TraceRow[]>("metrics_traces", {
      date,
      source,
      limit,
    });
    return NextResponse.json(rows, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: message },
      { status: daemonErrorStatus(err) },
    );
  }
}
