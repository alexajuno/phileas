import { SiteHeader } from "@/components/site-header";
import { ToolPlaygroundView } from "@/components/tool-playground-view";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <div className="mx-auto w-full max-w-3xl px-5 pb-16 sm:px-6">
      <SiteHeader currentTab="tools" />
      <ToolPlaygroundView />
    </div>
  );
}
