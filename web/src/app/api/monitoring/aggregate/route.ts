import { NextResponse, type NextRequest } from "next/server";

import { callDaemon, daemonErrorStatus } from "@/lib/daemon";
import type { AggregateResult } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const daysRaw = request.nextUrl.searchParams.get("days");
  const days = daysRaw ? Number.parseInt(daysRaw, 10) : 7;
  if (!Number.isFinite(days) || days <= 0 || days > 365) {
    return NextResponse.json({ error: "invalid days" }, { status: 400 });
  }
  try {
    const data = await callDaemon<AggregateResult>("metrics_aggregate", {
      days,
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
