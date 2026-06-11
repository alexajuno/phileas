import { NextResponse, type NextRequest } from "next/server";

import { isValidDay, localDayBoundsAsUtcIso, todayLocal } from "@/lib/day";
import { callDaemon, daemonErrorStatus } from "@/lib/daemon";
import type { MemoryItem } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const day = request.nextUrl.searchParams.get("date") ?? todayLocal();
  if (!isValidDay(day)) {
    return NextResponse.json({ error: "invalid date" }, { status: 400 });
  }
  try {
    const { startIso, endIso } = localDayBoundsAsUtcIso(day);
    const items = await callDaemon<MemoryItem[]>("memories_for_day", {
      start: startIso,
      end: endIso,
    });
    return NextResponse.json(items, {
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
