import { NextResponse, type NextRequest } from "next/server";

import { isValidDay } from "@/lib/day";
import { listTraces } from "@/lib/metrics-db";

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
    const rows = listTraces({ date, source, limit });
    return NextResponse.json(rows, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const status = message.includes("unable to open") ? 503 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
