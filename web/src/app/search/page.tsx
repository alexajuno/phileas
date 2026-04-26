import { SearchView } from "@/components/search-view";
import { SiteHeader } from "@/components/site-header";
import { searchMemories } from "@/lib/queries";

export const dynamic = "force-dynamic";

type SearchParams = Promise<{ q?: string | string[] }>;

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

  let initialItems: Awaited<ReturnType<typeof searchMemories>> = [];
  let error: string | null = null;
  if (q) {
    try {
      initialItems = searchMemories(q, 100);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-5 pb-16 sm:px-6">
      <SiteHeader />

      {error ? (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <p className="font-medium">Could not read memory.db</p>
          <p className="mt-1 font-mono text-xs opacity-80">{error}</p>
        </div>
      ) : (
        <SearchView initialQuery={q} initialItems={initialItems} />
      )}
    </div>
  );
}
