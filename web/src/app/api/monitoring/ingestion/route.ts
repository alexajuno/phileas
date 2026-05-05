import { NextResponse, type NextRequest } from "next/server";

import {
  fetchIngestionHealth,
  listIngestionEvents,
} from "@/lib/phileas-db";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const limitParam = searchParams.get("limit");

  let limit: number | undefined;
  if (limitParam != null) {
    const parsed = Number.parseInt(limitParam, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return NextResponse.json({ error: "invalid limit" }, { status: 400 });
    }
    limit = parsed;
  }

  try {
    const health = fetchIngestionHealth();
    const events = listIngestionEvents({ limit });
    return NextResponse.json(
      { health, events },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const code = message.includes("unable to open") ? 503 : 500;
    return NextResponse.json({ error: message }, { status: code });
  }
}
