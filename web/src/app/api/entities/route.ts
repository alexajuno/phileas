import { NextResponse, type NextRequest } from "next/server";

import {
  DaemonError,
  DaemonUnavailableError,
} from "@/lib/daemon";
import { listEntities } from "@/lib/graph";

export const dynamic = "force-dynamic";

const MAX_QUERY_LEN = 200;
const DEFAULT_LIMIT = 500;

export async function GET(request: NextRequest) {
  const sp = request.nextUrl.searchParams;
  const q = (sp.get("q") ?? "").trim().toLowerCase();
  if (q.length > MAX_QUERY_LEN) {
    return NextResponse.json(
      { error: `query too long (max ${MAX_QUERY_LEN})` },
      { status: 400 },
    );
  }
  const type = sp.get("type")?.trim() || null;

  const limitParam = sp.get("limit");
  const parsedLimit = limitParam ? Number.parseInt(limitParam, 10) : NaN;
  const limit =
    Number.isFinite(parsedLimit) && parsedLimit > 0 && parsedLimit <= 5000
      ? parsedLimit
      : DEFAULT_LIMIT;

  try {
    const items = await listEntities({ limit, type_filter: type });
    const filtered = q
      ? items.filter(
          (e) =>
            e.name.toLowerCase().includes(q) ||
            e.aliases.some((a) => a.toLowerCase().includes(q)),
        )
      : items;
    return NextResponse.json(filtered, {
      headers: { "Cache-Control": "no-store" },
    });
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
