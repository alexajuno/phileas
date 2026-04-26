import { EntityListView } from "@/components/entity-list-view";
import { SiteHeader } from "@/components/site-header";
import { DaemonUnavailableError } from "@/lib/daemon";
import { listEntities } from "@/lib/graph";
import type { EntitySummary } from "@/lib/types";

export const dynamic = "force-dynamic";

type SearchParams = Promise<{
  q?: string | string[];
  type?: string | string[];
}>;

function firstString(v: string | string[] | undefined): string | undefined {
  return Array.isArray(v) ? v[0] : v;
}

export default async function Page({
  searchParams,
}: {
  searchParams: SearchParams;
}) {
  const sp = await searchParams;
  const q = (firstString(sp.q) ?? "").trim();
  const type = firstString(sp.type)?.trim() || null;

  let initialItems: EntitySummary[] = [];
  let unavailable = false;
  let error: string | null = null;
  try {
    const all = await listEntities({ limit: 500, type_filter: type });
    initialItems = q
      ? all.filter(
          (e) =>
            e.name.toLowerCase().includes(q.toLowerCase()) ||
            e.aliases.some((a) => a.toLowerCase().includes(q.toLowerCase())),
        )
      : all;
  } catch (err) {
    if (err instanceof DaemonUnavailableError) {
      unavailable = true;
    } else {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-5 pb-16 sm:px-6">
      <SiteHeader currentTab="entities" />

      {error ? (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <p className="font-medium">Could not load entities</p>
          <p className="mt-1 font-mono text-xs opacity-80">{error}</p>
        </div>
      ) : (
        <EntityListView
          initialQuery={q}
          initialType={type}
          initialItems={initialItems}
          unavailable={unavailable}
        />
      )}
    </div>
  );
}
