import { NextResponse, type NextRequest } from "next/server";

import { searchMemories } from "@/lib/queries";

export const dynamic = "force-dynamic";

const MAX_QUERY_LEN = 200;

export async function GET(request: NextRequest) {
  const q = (request.nextUrl.searchParams.get("q") ?? "").trim();
  if (!q) {
    return NextResponse.json([], {
      headers: { "Cache-Control": "no-store" },
    });
  }
  if (q.length > MAX_QUERY_LEN) {
    return NextResponse.json(
      { error: `query too long (max ${MAX_QUERY_LEN})` },
      { status: 400 },
    );
  }

  const limitParam = request.nextUrl.searchParams.get("limit");
  const parsedLimit = limitParam ? Number.parseInt(limitParam, 10) : NaN;
  const limit =
    Number.isFinite(parsedLimit) && parsedLimit > 0 && parsedLimit <= 500
      ? parsedLimit
      : 100;

  try {
    const items = searchMemories(q, limit);
    return NextResponse.json(items, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const status = message.includes("unable to open") ? 503 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
