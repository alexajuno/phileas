import { NextResponse } from "next/server";

import { fetchDaysWithCounts } from "@/lib/queries";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const days = fetchDaysWithCounts(60);
    return NextResponse.json(days, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const status = message.includes("unable to open") ? 503 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
