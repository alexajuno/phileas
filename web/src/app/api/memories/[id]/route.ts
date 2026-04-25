import { NextResponse } from "next/server";

import {
  DaemonError,
  DaemonUnavailableError,
  callDaemon,
} from "@/lib/daemon";

export const dynamic = "force-dynamic";

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  if (!id || typeof id !== "string") {
    return NextResponse.json({ error: "missing id" }, { status: 400 });
  }

  try {
    const result = await callDaemon<string>("forget", {
      memory_id: id,
      reason: "web-ui",
    });
    return NextResponse.json({ ok: true, result });
  } catch (err) {
    if (err instanceof DaemonUnavailableError) {
      return NextResponse.json({ error: err.message }, { status: 503 });
    }
    if (err instanceof DaemonError) {
      return NextResponse.json({ error: err.message }, { status: 500 });
    }
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
