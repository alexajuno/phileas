import { NextResponse, type NextRequest } from "next/server";

import { aggregateRecent } from "@/lib/metrics-db";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const daysRaw = request.nextUrl.searchParams.get("days");
  const days = daysRaw ? Number.parseInt(daysRaw, 10) : 7;
  if (!Number.isFinite(days) || days <= 0 || days > 365) {
    return NextResponse.json({ error: "invalid days" }, { status: 400 });
  }
  try {
    const data = aggregateRecent(days);
    return NextResponse.json(data, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const status = message.includes("unable to open") ? 503 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
