import { NextResponse } from "next/server";

import { callDaemon, daemonErrorStatus } from "@/lib/daemon";
import type { DayCount } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const days = await callDaemon<DayCount[]>("memories_days", {
      limit: 60,
      tz_offset_minutes: -new Date().getTimezoneOffset(),
    });
    return NextResponse.json(days, {
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
