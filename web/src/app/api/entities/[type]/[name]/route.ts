import { NextResponse } from "next/server";

import {
  DaemonError,
  DaemonUnavailableError,
} from "@/lib/daemon";
import {
  findEntity,
  getEntityMemoryIds,
  getEntityRelations,
} from "@/lib/graph";
import { fetchMemoriesByIds } from "@/lib/queries";
import type { EntityDetail } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ type: string; name: string }> },
) {
  const { type: rawType, name: rawName } = await params;
  const type = decodeURIComponent(rawType);
  const name = decodeURIComponent(rawName);
  if (!type || !name) {
    return NextResponse.json({ error: "missing type or name" }, { status: 400 });
  }

  try {
    const node = await findEntity(type, name);
    if (!node) {
      return NextResponse.json({ error: "entity not found" }, { status: 404 });
    }
    const [relations, memoryIds] = await Promise.all([
      getEntityRelations(type, name),
      getEntityMemoryIds(type, name),
    ]);
    let memories: EntityDetail["memories"] = [];
    try {
      memories = fetchMemoriesByIds(memoryIds);
    } catch {
      // SQLite read failure shouldn't blank the whole detail page.
      memories = [];
    }
    const detail: EntityDetail = {
      name: node.name,
      type: node.type,
      aliases: node.aliases,
      props: node.props,
      relations,
      memories,
    };
    return NextResponse.json(detail, {
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
