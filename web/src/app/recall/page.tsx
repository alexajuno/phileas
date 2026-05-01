import { RecallView } from "@/components/recall-view";
import { SiteHeader } from "@/components/site-header";

export const dynamic = "force-dynamic";

type SearchParams = Promise<{
  q?: string | string[];
  k?: string | string[];
  t?: string | string[];
  m?: string | string[];
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
  const mRaw = Number.parseInt(firstString(sp.m) ?? "", 10);
  const m = Number.isFinite(mRaw) && mRaw >= 0 && mRaw <= 10 ? mRaw : 0;

  return (
    <div className="mx-auto w-full max-w-3xl px-5 pb-16 sm:px-6">
      <SiteHeader currentTab="playground" />
      <RecallView
        initialQuery={q}
        initialTopK={k}
        initialType={t}
        initialMinImportance={m}
      />
    </div>
  );
}
