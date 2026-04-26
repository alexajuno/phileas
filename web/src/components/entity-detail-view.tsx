"use client";

import { useMemo } from "react";
import Link from "next/link";
import { ArrowLeft, ArrowRight } from "lucide-react";

import { Badge } from "./ui/badge";
import { MemoryCard } from "./memory-card";
import { localDayOf } from "@/lib/day";
import type { EntityDetail, EntityRelation } from "@/lib/types";
import { cn } from "@/lib/utils";

type Props = {
  entity: EntityDetail;
};

function relationKey(r: EntityRelation): string {
  return `${r.direction}:${r.edge_type}:${r.type}:${r.name}`;
}

export function EntityDetailView({ entity }: Props) {
  const groupedRelations = useMemo(() => {
    const groups = new Map<string, EntityRelation[]>();
    for (const r of entity.relations) {
      const bucket = groups.get(r.edge_type) ?? [];
      bucket.push(r);
      groups.set(r.edge_type, bucket);
    }
    return [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [entity.relations]);

  const propEntries = useMemo(
    () => Object.entries(entity.props).filter(([, v]) => v !== null && v !== ""),
    [entity.props],
  );

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="outline">{entity.type}</Badge>
          <h2 className="text-lg font-medium tracking-tight">{entity.name}</h2>
        </div>
        {entity.aliases.length > 0 && (
          <p className="text-xs text-muted-foreground">
            also known as {entity.aliases.join(" · ")}
          </p>
        )}
        {propEntries.length > 0 && (
          <dl className="mt-3 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-xs">
            {propEntries.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-muted-foreground">{k}</dt>
                <dd className="text-foreground/90">{String(v)}</dd>
              </div>
            ))}
          </dl>
        )}
      </section>

      <section className="space-y-3">
        <h3 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Relationships
        </h3>
        {groupedRelations.length === 0 ? (
          <p className="text-xs text-muted-foreground">No relationships yet.</p>
        ) : (
          <ul className="space-y-2">
            {groupedRelations.map(([edgeType, rels]) => (
              <li
                key={edgeType}
                className="rounded-lg border border-border/60 bg-card/40 px-3 py-2"
              >
                <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                  {edgeType}
                </div>
                <ul className="space-y-1">
                  {rels.map((r) => (
                    <li
                      key={relationKey(r)}
                      className="flex items-center gap-2 text-sm"
                    >
                      {r.direction === "out" ? (
                        <ArrowRight
                          aria-hidden
                          className="h-3 w-3 shrink-0 text-muted-foreground"
                        />
                      ) : (
                        <ArrowLeft
                          aria-hidden
                          className="h-3 w-3 shrink-0 text-muted-foreground"
                        />
                      )}
                      <Badge variant="outline" className="shrink-0">
                        {r.type}
                      </Badge>
                      <Link
                        href={`/entities/${encodeURIComponent(r.type)}/${encodeURIComponent(r.name)}`}
                        className={cn(
                          "truncate text-foreground transition-colors",
                          "hover:text-foreground/70 hover:underline",
                        )}
                      >
                        {r.name}
                      </Link>
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="space-y-3">
        <h3 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Memories ({entity.memories.length})
        </h3>
        {entity.memories.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No memories link to this entity.
          </p>
        ) : (
          <ul className="space-y-2.5">
            {entity.memories.map((m) => (
              <li key={m.id}>
                <MemoryCard memory={m} dayBadge={localDayOf(m.created_at)} />
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
