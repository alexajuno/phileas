"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Search, X } from "lucide-react";

import { MemoryCard } from "./memory-card";
import { localDayOf } from "@/lib/day";
import { tokenizeQuery } from "@/lib/highlight";
import type { MemoryItem } from "@/lib/types";
import { cn } from "@/lib/utils";

type Props = {
  initialQuery: string;
  initialItems: MemoryItem[];
};

const DEBOUNCE_MS = 250;

export function SearchView({ initialQuery, initialItems }: Props) {
  const [query, setQuery] = useState(initialQuery);
  const [items, setItems] = useState<MemoryItem[]>(initialItems);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(initialQuery.length > 0);
  const reduceMotion = useReducedMotion();
  const inputRef = useRef<HTMLInputElement>(null);
  const reqIdRef = useRef(0);

  // Auto-focus the input on mount, place caret at end.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.focus();
    el.setSelectionRange(el.value.length, el.value.length);
  }, []);

  const syncUrl = useCallback((q: string) => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    const search = params.toString();
    const next = `${window.location.pathname}${search ? `?${search}` : ""}`;
    window.history.replaceState(null, "", next);
  }, []);

  const runSearch = useCallback(async (q: string) => {
    const trimmed = q.trim();
    const reqId = ++reqIdRef.current;
    if (!trimmed) {
      setItems([]);
      setHasSearched(false);
      setError(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(trimmed)}`, {
        cache: "no-store",
      });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? `HTTP ${res.status}`);
      }
      const data = (await res.json()) as MemoryItem[];
      if (reqId !== reqIdRef.current) return; // stale
      setItems(data);
      setError(null);
      setHasSearched(true);
    } catch (err) {
      if (reqId !== reqIdRef.current) return;
      setError(err instanceof Error ? err.message : String(err));
      setHasSearched(true);
    } finally {
      if (reqId === reqIdRef.current) setLoading(false);
    }
  }, []);

  // Debounced fetch on query change. Skip the very first effect run when SSR
  // already returned matching initialItems for initialQuery.
  const skipFirstFetchRef = useRef(initialQuery.length > 0);
  useEffect(() => {
    if (skipFirstFetchRef.current) {
      skipFirstFetchRef.current = false;
      syncUrl(query);
      return;
    }
    syncUrl(query);
    const handle = window.setTimeout(() => {
      runSearch(query);
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
  }, [query, runSearch, syncUrl]);

  const terms = useMemo(() => tokenizeQuery(query), [query]);
  const trimmed = query.trim();

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
          placeholder="Search memories…"
          autoComplete="off"
          spellCheck={false}
          aria-label="Search memories"
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
            aria-label="Clear search"
            className={cn(
              "absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1",
              "text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground",
            )}
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
        <span>
          {loading
            ? "searching…"
            : hasSearched && trimmed
              ? `${items.length} match${items.length === 1 ? "" : "es"} for "${trimmed}"`
              : "Type a term to search across summaries, raw text, and tags."}
        </span>
        {hasSearched && items.length >= 100 && !loading && (
          <span className="tabular-nums">showing first 100</span>
        )}
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      {!hasSearched && !trimmed ? null : items.length === 0 && !loading && !error ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full border border-dashed border-border/70 text-muted-foreground">
            <Search className="h-5 w-5" aria-hidden />
          </div>
          <p className="text-sm text-foreground/90">No matches.</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Try a shorter or different term.
          </p>
        </div>
      ) : (
        <ul className="space-y-2.5">
          <AnimatePresence initial={false}>
            {items.map((m, i) => (
              <motion.li
                key={m.id}
                layout
                initial={reduceMotion ? false : { opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: -4 }}
                transition={{
                  duration: 0.2,
                  ease: [0.22, 0.61, 0.36, 1],
                  delay: reduceMotion ? 0 : Math.min(i, 12) * 0.02,
                }}
              >
                <MemoryCard
                  memory={m}
                  highlightTerms={terms}
                  dayBadge={localDayOf(m.created_at)}
                />
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </div>
  );
}
