"use client";

import { useEffect, useRef, useState } from "react";
import { Download } from "lucide-react";

type Props = {
  day: string;
  type: string | null;
  min: number;
};

export function ExportMenu({ day, type, min }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onClick(e: MouseEvent) {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("mousedown", onClick);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onClick);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  function buildHref(format: "json" | "markdown"): string {
    const params = new URLSearchParams({ format, from: day, to: day });
    if (type) params.set("type", type);
    if (min > 1) params.set("min", String(min));
    return `/api/export?${params.toString()}`;
  }

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 rounded-md border border-border/60 bg-card/60 px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:border-border hover:text-foreground"
        title="Export this day's memories"
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <Download className="h-3 w-3" aria-hidden />
        export
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 z-20 mt-1 w-32 overflow-hidden rounded-md border border-border/60 bg-popover text-xs shadow-md"
        >
          <a
            role="menuitem"
            href={buildHref("json")}
            download
            onClick={() => setOpen(false)}
            className="block px-3 py-1.5 text-foreground/90 hover:bg-accent hover:text-accent-foreground"
          >
            JSON
          </a>
          <a
            role="menuitem"
            href={buildHref("markdown")}
            download
            onClick={() => setOpen(false)}
            className="block px-3 py-1.5 text-foreground/90 hover:bg-accent hover:text-accent-foreground"
          >
            Markdown
          </a>
        </div>
      )}
    </div>
  );
}
