import "server-only";

import { callDaemon } from "./daemon";
import type { EntityRelation, EntitySummary } from "./types";

type RawEntity = {
  name: string;
  type: string;
  aliases?: string;
  memory_count?: number;
};

type RawNode = {
  name: string;
  type: string;
  props?: string;
  aliases?: string;
};

function parseAliases(raw: string | undefined): string[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}

function parseProps(raw: string | undefined): Record<string, unknown> {
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

export async function listEntities(opts: {
  limit?: number;
  type_filter?: string | null;
} = {}): Promise<EntitySummary[]> {
  const params: Record<string, unknown> = {
    op: "list_all_entities",
    limit: opts.limit ?? 500,
  };
  if (opts.type_filter) params.type_filter = opts.type_filter;
  const rows = await callDaemon<RawEntity[]>("graph_read", params);
  return (rows ?? []).map((r) => ({
    name: r.name,
    type: r.type,
    aliases: parseAliases(r.aliases),
    memory_count: r.memory_count ?? 0,
  }));
}

export async function findEntity(
  type: string,
  name: string,
): Promise<{
  name: string;
  type: string;
  props: Record<string, unknown>;
  aliases: string[];
} | null> {
  const rows = await callDaemon<RawNode[]>("graph_read", {
    op: "find_nodes",
    node_type: type,
    name,
  });
  const first = (rows ?? [])[0];
  if (!first) return null;
  return {
    name: first.name,
    type: first.type,
    props: parseProps(first.props),
    aliases: parseAliases(first.aliases),
  };
}

export async function getEntityRelations(
  type: string,
  name: string,
): Promise<EntityRelation[]> {
  const rows = await callDaemon<EntityRelation[]>("graph_read", {
    op: "get_related_entities",
    entity_type: type,
    entity_name: name,
  });
  return rows ?? [];
}

export async function getEntityMemoryIds(
  type: string,
  name: string,
): Promise<string[]> {
  const rows = await callDaemon<string[]>("graph_read", {
    op: "get_memories_about",
    entity_type: type,
    entity_name: name,
  });
  return rows ?? [];
}
