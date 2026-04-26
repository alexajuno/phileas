"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Search, X } from "lucide-react";

import { Badge } from "./ui/badge";
import type { EntitySummary } from "@/lib/types";
import { cn } from "@/lib/utils";

type Props = {
  initialQuery: string;
  initialType: string | null;
  initialItems: EntitySummary[];
  unavailable?: boolean;
};

const DEBOUNCE_MS = 250;

function entityHref(e: EntitySummary): string {
  return `/entities/${encodeURIComponent(e.type)}/${encodeURIComponent(e.name)}`;
}

export function EntityListView({
  initialQuery,
  initialType,
  initialItems,
  unavailable = false,
}: Props) {
  const [query, setQuery] = useState(initialQuery);
  const [type, setType] = useState<string | null>(initialType);
  const [items, setItems] = useState<EntitySummary[]>(initialItems);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reduceMotion = useReducedMotion();
  const inputRef = useRef<HTMLInputElement>(null);
  const reqIdRef = useRef(0);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const syncUrl = useCallback((q: string, t: string | null) => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    if (t) params.set("type", t);
    const search = params.toString();
    const next = `${window.location.pathname}${search ? `?${search}` : ""}`;
    window.history.replaceState(null, "", next);
  }, []);

  const fetchEntities = useCallback(async (q: string, t: string | null) => {
    const reqId = ++reqIdRef.current;
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (q.trim()) params.set("q", q.trim());
      if (t) params.set("type", t);
      const url = `/api/entities${params.toString() ? `?${params}` : ""}`;
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? `HTTP ${res.status}`);
      }
      const data = (await res.json()) as EntitySummary[];
      if (reqId !== reqIdRef.current) return;
      setItems(data);
      setError(null);
    } catch (err) {
      if (reqId !== reqIdRef.current) return;
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (reqId === reqIdRef.current) setLoading(false);
    }
  }, []);

  const skipFirstFetchRef = useRef(true);
  useEffect(() => {
    if (skipFirstFetchRef.current) {
      skipFirstFetchRef.current = false;
      syncUrl(query, type);
      return;
    }
    syncUrl(query, type);
    const handle = window.setTimeout(() => {
      fetchEntities(query, type);
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
  }, [query, type, fetchEntities, syncUrl]);

  const types = useMemo(() => {
    const counts = new Map<string, number>();
    for (const e of items) counts.set(e.type, (counts.get(e.type) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [items]);

  if (unavailable) {
    return (
      <div className="rounded-lg border border-border/60 bg-muted/30 px-4 py-8 text-center text-sm text-muted-foreground">
        <p className="font-medium text-foreground/90">Graph unavailable</p>
        <p className="mt-1 text-xs">
          The Phileas daemon isn&apos;t running, so entities can&apos;t be loaded right now.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div className="relative">
        <Search
          aria-hidden
          className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
        />
        <input
          ref={inputRef}
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter entities…"
          autoComplete="off"
          spellCheck={false}
          aria-label="Filter entities"
          className={cn(
            "w-full rounded-xl border border-border/60 bg-card/60",
            "py-2.5 pl-9 pr-9 text-sm text-foreground placeholder:text-muted-foreground/70",
            "outline-none transition-colors",
            "hover:border-border focus:border-foreground/40 focus:bg-card/80",
          )}
        />
        {query && (
          <button
            type="button"
            onClick={() => {
              setQuery("");
              inputRef.current?.focus();
            }}
            aria-label="Clear filter"
            className={cn(
              "absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1",
              "text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground",
            )}
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {types.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setType(null)}
            className={cn(
              "rounded-full border px-2.5 py-0.5 text-[11px] transition-colors",
              type === null
                ? "border-foreground/40 bg-foreground/10 text-foreground"
                : "border-border/60 text-muted-foreground hover:border-border hover:text-foreground",
            )}
          >
            all
          </button>
          {types.map(([t, count]) => (
            <button
              key={t}
              type="button"
              onClick={() => setType(t === type ? null : t)}
              className={cn(
                "rounded-full border px-2.5 py-0.5 text-[11px] transition-colors",
                type === t
                  ? "border-foreground/40 bg-foreground/10 text-foreground"
                  : "border-border/60 text-muted-foreground hover:border-border hover:text-foreground",
              )}
            >
              {t} <span className="tabular-nums opacity-60">{count}</span>
            </button>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
        <span>
          {loading
            ? "loading…"
            : `${items.length} entit${items.length === 1 ? "y" : "ies"}`}
        </span>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      {items.length === 0 && !loading && !error ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full border border-dashed border-border/70 text-muted-foreground">
            <Search className="h-5 w-5" aria-hidden />
          </div>
          <p className="text-sm text-foreground/90">No entities.</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Try a different filter.
          </p>
        </div>
      ) : (
        <ul className="divide-y divide-border/60 rounded-xl border border-border/60 bg-card/40">
          <AnimatePresence initial={false}>
            {items.map((e, i) => (
              <motion.li
                key={`${e.type}:${e.name}`}
                layout
                initial={reduceMotion ? false : { opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: -4 }}
                transition={{
                  duration: 0.18,
                  ease: [0.22, 0.61, 0.36, 1],
                  delay: reduceMotion ? 0 : Math.min(i, 12) * 0.015,
                }}
              >
                <Link
                  href={entityHref(e)}
                  className="flex items-center justify-between gap-3 px-4 py-2.5 text-sm transition-colors hover:bg-muted/40"
                >
                  <div className="flex min-w-0 items-center gap-2.5">
                    <Badge variant="outline" className="shrink-0">
                      {e.type}
                    </Badge>
                    <span className="truncate text-foreground">{e.name}</span>
                    {e.aliases.length > 0 && (
                      <span className="truncate text-xs text-muted-foreground">
                        {e.aliases.join(" · ")}
                      </span>
                    )}
                  </div>
                  <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                    {e.memory_count}
                  </span>
                </Link>
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </div>
  );
}
