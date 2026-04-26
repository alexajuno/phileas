import Link from "next/link";
import { notFound } from "next/navigation";

import { EntityDetailView } from "@/components/entity-detail-view";
import { ThemeToggle } from "@/components/theme-toggle";
import { DaemonUnavailableError } from "@/lib/daemon";
import {
  findEntity,
  getEntityMemoryIds,
  getEntityRelations,
} from "@/lib/graph";
import { fetchMemoriesByIds } from "@/lib/queries";
import type { EntityDetail } from "@/lib/types";

export const dynamic = "force-dynamic";

type Params = Promise<{ type: string; name: string }>;

type LoadResult =
  | { kind: "ok"; detail: EntityDetail }
  | { kind: "missing" }
  | { kind: "unavailable" }
  | { kind: "error"; message: string };

async function loadEntity(type: string, name: string): Promise<LoadResult> {
  let node: Awaited<ReturnType<typeof findEntity>>;
  let relations: Awaited<ReturnType<typeof getEntityRelations>>;
  let memoryIds: string[];
  try {
    node = await findEntity(type, name);
    if (!node) return { kind: "missing" };
    [relations, memoryIds] = await Promise.all([
      getEntityRelations(type, name),
      getEntityMemoryIds(type, name),
    ]);
  } catch (err) {
    if (err instanceof DaemonUnavailableError) return { kind: "unavailable" };
    return { kind: "error", message: err instanceof Error ? err.message : String(err) };
  }
  let memories: EntityDetail["memories"] = [];
  try {
    memories = fetchMemoriesByIds(memoryIds);
  } catch {
    memories = [];
  }
  return {
    kind: "ok",
    detail: {
      name: node.name,
      type: node.type,
      aliases: node.aliases,
      props: node.props,
      relations,
      memories,
    },
  };
}

export default async function Page({ params }: { params: Params }) {
  const { type: rawType, name: rawName } = await params;
  const type = decodeURIComponent(rawType);
  const name = decodeURIComponent(rawName);

  const result = await loadEntity(type, name);
  if (result.kind === "missing") notFound();

  const detail = result.kind === "ok" ? result.detail : null;
  const unavailable = result.kind === "unavailable";
  const error = result.kind === "error" ? result.message : null;

  return (
    <div className="mx-auto w-full max-w-3xl px-5 pb-16 pt-6 sm:px-6">
      <header className="mb-6 flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-base font-medium tracking-tight">
            Phileas <span className="text-muted-foreground">· entity</span>
          </h1>
          <p className="mt-1 text-xs text-muted-foreground">
            {type} · {name}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <ThemeToggle />
          <Link
            href="/entities"
            className="text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            ← entities
          </Link>
        </div>
      </header>

      {unavailable ? (
        <div className="rounded-lg border border-border/60 bg-muted/30 px-4 py-8 text-center text-sm text-muted-foreground">
          <p className="font-medium text-foreground/90">Graph unavailable</p>
          <p className="mt-1 text-xs">
            The Phileas daemon isn&apos;t running, so this entity can&apos;t be loaded right now.
          </p>
        </div>
      ) : error ? (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <p className="font-medium">Could not load entity</p>
          <p className="mt-1 font-mono text-xs opacity-80">{error}</p>
        </div>
      ) : detail ? (
        <EntityDetailView entity={detail} />
      ) : null}
    </div>
  );
}
