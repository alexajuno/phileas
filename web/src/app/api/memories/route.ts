import { NextResponse, type NextRequest } from "next/server";

import { isValidDay, todayLocal } from "@/lib/day";
import { fetchMemoriesForDay } from "@/lib/queries";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const day = request.nextUrl.searchParams.get("date") ?? todayLocal();
  if (!isValidDay(day)) {
    return NextResponse.json({ error: "invalid date" }, { status: 400 });
  }
  try {
    const items = fetchMemoriesForDay(day);
    return NextResponse.json(items, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const status = message.includes("unable to open") ? 503 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
