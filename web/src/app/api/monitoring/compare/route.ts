import { NextResponse, type NextRequest } from "next/server";

import { callDaemon, daemonErrorStatus } from "@/lib/daemon";
import type { CompareResult } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const cutoff = searchParams.get("cutoff");
  if (!cutoff) {
    return NextResponse.json({ error: "cutoff required (ISO timestamp)" }, { status: 400 });
  }
  const cutoffDate = new Date(cutoff);
  if (Number.isNaN(cutoffDate.getTime())) {
    return NextResponse.json({ error: "invalid cutoff" }, { status: 400 });
  }
  const source = searchParams.get("source") ?? undefined;
  const daysRaw = searchParams.get("days");
  const windowDays = daysRaw ? Number.parseInt(daysRaw, 10) : 7;
  if (!Number.isFinite(windowDays) || windowDays <= 0 || windowDays > 365) {
    return NextResponse.json({ error: "invalid days" }, { status: 400 });
  }
  try {
    const data = await callDaemon<CompareResult>("metrics_compare", {
      cutoff: cutoffDate.toISOString(),
      source,
      window_days: windowDays,
    });
    return NextResponse.json(data, {
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
