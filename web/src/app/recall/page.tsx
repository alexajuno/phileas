import { RecallView } from "@/components/recall-view";
import { SiteHeader } from "@/components/site-header";

export const dynamic = "force-dynamic";

type SearchParams = Promise<{
  q?: string | string[];
  k?: string | string[];
  t?: string | string[];
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
  const kRaw = Number.parseInt(firstString(sp.k) ?? "", 10);
  const k =
    Number.isFinite(kRaw) && kRaw >= 1 && kRaw <= 100 ? kRaw : 30;
  const t = firstString(sp.t) ?? "";

  return (
    <div className="mx-auto w-full max-w-3xl px-5 pb-16 sm:px-6">
      <SiteHeader currentTab="playground" />
      <RecallView initialQuery={q} initialTopK={k} initialType={t} />
    </div>
  );
}
