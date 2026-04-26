"use client";

import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

const POLL_MS = 30_000;

export function DaemonStatus() {
  const [ok, setOk] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await fetch("/api/daemon/status", { cache: "no-store" });
        if (cancelled) return;
        const body = (await res.json().catch(() => ({}))) as { ok?: boolean };
        setOk(Boolean(body.ok));
      } catch {
        if (!cancelled) setOk(false);
      }
    };
    tick();
    const id = window.setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const label =
    ok === null ? "checking daemon…" : ok ? "daemon up" : "daemon down";

  return (
    <span
      role="status"
      aria-label={label}
      title={label}
      className={cn(
        "inline-block h-1.5 w-1.5 rounded-full transition-colors",
        ok === null && "bg-muted-foreground/40",
        ok === true && "bg-emerald-500 ring-1 ring-emerald-500/30",
        ok === false && "bg-red-500 ring-1 ring-red-500/30",
      )}
    />
  );
}
