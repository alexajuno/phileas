"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";

import { DayNav } from "./day-nav";
import { EmptyState } from "./empty-state";
import { MemoryCard } from "./memory-card";
import { StatsStrip } from "./stats-strip";
import { todayLocal } from "@/lib/day";
import type { MemoryItem } from "@/lib/types";

type Props = {
  initialDay: string;
  initialItems: MemoryItem[];
};

const POLL_MS = 20_000;

export function MemoryList({ initialDay, initialItems }: Props) {
  const [day, setDay] = useState(initialDay);
  const [items, setItems] = useState<MemoryItem[]>(initialItems);
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastLoaded, setLastLoaded] = useState<Date | null>(null);
  const [loading, setLoading] = useState(false);
  const [justArrived, setJustArrived] = useState<Set<string>>(() => new Set());
  const prevIdsRef = useRef<Set<string>>(new Set(initialItems.map((m) => m.id)));
  const reduceMotion = useReducedMotion();

  const today = todayLocal();
  const isToday = day === today;

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
    setSelectedType(null);
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
    () =>
      selectedType
        ? items.filter((m) => m.memory_type === selectedType)
        : items,
    [items, selectedType],
  );

  // Clear the filter automatically if no items match (e.g. after a reload).
  useEffect(() => {
    if (selectedType && visibleItems.length === 0) {
      setSelectedType(null);
    }
  }, [selectedType, visibleItems.length]);

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
                />
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </div>
  );
}
