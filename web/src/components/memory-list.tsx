"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";

import { DayNav } from "./day-nav";
import { EmptyState } from "./empty-state";
import { ExportMenu } from "./export-menu";
import { MemoryCard } from "./memory-card";
import { StatsStrip } from "./stats-strip";
import { todayLocal } from "@/lib/day";
import type { MemoryItem } from "@/lib/types";

type Props = {
  initialDay: string;
  initialItems: MemoryItem[];
  initialType: string | null;
};

const POLL_MS = 20_000;

function buildQuery({
  day,
  type,
}: {
  day: string;
  type: string | null;
}): string {
  const params = new URLSearchParams();
  if (day !== todayLocal()) params.set("day", day);
  if (type) params.set("type", type);
  const q = params.toString();
  return q ? `?${q}` : window.location.pathname;
}

export function MemoryList({
  initialDay,
  initialItems,
  initialType,
}: Props) {
  const [day, setDayState] = useState(initialDay);
  const [items, setItems] = useState<MemoryItem[]>(initialItems);
  const [selectedType, setSelectedTypeState] = useState<string | null>(initialType);
  const [error, setError] = useState<string | null>(null);
  const [lastLoaded, setLastLoaded] = useState<Date | null>(null);
  const [loading, setLoading] = useState(false);
  const [justArrived, setJustArrived] = useState<Set<string>>(() => new Set());
  const prevIdsRef = useRef<Set<string>>(new Set(initialItems.map((m) => m.id)));
  const reduceMotion = useReducedMotion();

  const today = todayLocal();
  const isToday = day === today;

  const syncUrl = useCallback(
    (next: { day: string; type: string | null }) => {
      if (typeof window === "undefined") return;
      window.history.replaceState(null, "", buildQuery(next));
    },
    [],
  );

  const setDay = useCallback(
    (d: string) => {
      setDayState(d);
      setSelectedTypeState(null);
      syncUrl({ day: d, type: null });
    },
    [syncUrl],
  );

  const setSelectedType = useCallback(
    (t: string | null) => {
      setSelectedTypeState(t);
      syncUrl({ day, type: t });
    },
    [day, syncUrl],
  );

  const load = useCallback(
    async (
      target: string,
      { silent = false, detectArrivals = false } = {},
    ) => {
      if (!silent) setLoading(true);
      try {
        const res = await fetch(
          `/api/memories?date=${encodeURIComponent(target)}`,
          { cache: "no-store" },
        );
        if (!res.ok) {
          const body = (await res.json().catch(() => ({}))) as { error?: string };
          throw new Error(body.error ?? `HTTP ${res.status}`);
        }
        const data = (await res.json()) as MemoryItem[];
        const nextIds = new Set(data.map((m) => m.id));
        if (detectArrivals) {
          const arrivals = new Set<string>();
          for (const id of nextIds) {
            if (!prevIdsRef.current.has(id)) arrivals.add(id);
          }
          if (arrivals.size) {
            setJustArrived(arrivals);
            setTimeout(() => setJustArrived(new Set()), 2400);
          }
        } else {
          setJustArrived(new Set());
        }
        prevIdsRef.current = nextIds;
        setItems(data);
        setError(null);
        setLastLoaded(new Date());
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!silent) setLoading(false);
      }
    },
    [],
  );

  // Reload when day changes (but not on initial mount — we already have SSR data).
  const mountedRef = useRef(false);
  useEffect(() => {
    if (!mountedRef.current) {
      mountedRef.current = true;
      setLastLoaded(new Date());
      return;
    }
    load(day);
  }, [day, load]);

  // Poll only when viewing today; detect arrivals so new items get highlighted.
  useEffect(() => {
    if (!isToday) return;
    const id = window.setInterval(
      () => load(day, { silent: true, detectArrivals: true }),
      POLL_MS,
    );
    return () => window.clearInterval(id);
  }, [isToday, day, load]);

  // Refresh on window focus when viewing today.
  useEffect(() => {
    if (!isToday) return;
    function onFocus() {
      load(day, { silent: true, detectArrivals: true });
    }
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [isToday, day, load]);

  const visibleItems = useMemo(
    () => items.filter((m) => !selectedType || m.memory_type === selectedType),
    [items, selectedType],
  );

  const handleForgotten = useCallback((id: string) => {
    setItems((prev) => prev.filter((m) => m.id !== id));
    prevIdsRef.current.delete(id);
    setJustArrived((prev) => {
      if (!prev.has(id)) return prev;
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  }, []);

  const clearFilters = useCallback(() => {
    setSelectedTypeState(null);
    syncUrl({ day, type: null });
  }, [day, syncUrl]);

  const filtersActive = selectedType !== null;
  const filtersHideAll =
    items.length > 0 && visibleItems.length === 0 && filtersActive;

  const lastLoadedLabel = useMemo(
    () =>
      lastLoaded
        ? lastLoaded.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          })
        : "",
    [lastLoaded],
  );

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <DayNav day={day} onChange={setDay} />
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
          <ExportMenu day={day} type={selectedType} />
          {loading && <span className="animate-pulse">loading…</span>}
          {!loading && isToday && lastLoaded && (
            <span className="inline-flex items-center gap-1.5">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400/60" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-400" />
              </span>
              live · {lastLoadedLabel}
            </span>
          )}
          {!loading && !isToday && lastLoaded && (
            <span className="tabular-nums">loaded {lastLoadedLabel}</span>
          )}
        </div>
      </div>

      <StatsStrip
        items={items}
        selectedType={selectedType}
        onSelect={setSelectedType}
      />

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      {items.length === 0 && !error ? (
        <EmptyState day={day} isToday={isToday} />
      ) : filtersHideAll ? (
        <div className="rounded-lg border border-border/60 bg-muted/30 px-3 py-3 text-xs text-muted-foreground">
          No memories match these filters.{" "}
          <button
            type="button"
            onClick={clearFilters}
            className="font-medium text-foreground underline-offset-2 hover:underline"
          >
            Clear filters
          </button>
        </div>
      ) : (
        <ul className="space-y-2.5">
          <AnimatePresence initial={false}>
            {visibleItems.map((m, i) => (
              <motion.li
                key={m.id}
                layout
                initial={reduceMotion ? false : { opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: -4 }}
                transition={{
                  duration: 0.24,
                  ease: [0.22, 0.61, 0.36, 1],
                  delay: reduceMotion ? 0 : Math.min(i, 12) * 0.03,
                }}
              >
                <MemoryCard
                  memory={m}
                  justArrived={justArrived.has(m.id)}
                  onForgotten={handleForgotten}
                />
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </div>
  );
}
