import { NextResponse } from "next/server";

import { callDaemon, DaemonUnavailableError } from "@/lib/daemon";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    await callDaemon("status", {});
    return NextResponse.json({ ok: true });
  } catch (err) {
    if (err instanceof DaemonUnavailableError) {
      return NextResponse.json({ ok: false });
    }
    return NextResponse.json({
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}
