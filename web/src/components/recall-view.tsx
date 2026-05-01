"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Sparkles } from "lucide-react";

import { MemoryCard } from "./memory-card";
import { localDayOf } from "@/lib/day";
import { tokenizeQuery } from "@/lib/highlight";
import type { RecallResult } from "@/app/api/recall/route";
import { cn } from "@/lib/utils";

type Props = {
  initialQuery: string;
  initialTopK: number;
  initialType: string;
  initialMinImportance: number;
};

const TOP_K_MIN = 1;
const TOP_K_MAX = 100;
const TOP_K_DEFAULT = 30;
const IMP_MAX = 10;

const TYPE_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "any type" },
  { value: "profile", label: "profile" },
  { value: "event", label: "event" },
  { value: "knowledge", label: "knowledge" },
  { value: "behavior", label: "behavior" },
  { value: "reflection", label: "reflection" },
  { value: "feedback", label: "feedback" },
  { value: "preference", label: "preference" },
  { value: "observation", label: "observation" },
  { value: "project", label: "project" },
  { value: "reference", label: "reference" },
];

export function RecallView({
  initialQuery,
  initialTopK,
  initialType,
  initialMinImportance,
}: Props) {
  const [query, setQuery] = useState(initialQuery);
  const [topK, setTopK] = useState(initialTopK);
  const [memoryType, setMemoryType] = useState(initialType);
  const [minImportance, setMinImportance] = useState(initialMinImportance);
  const [items, setItems] = useState<RecallResult[]>([]);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasRun, setHasRun] = useState(false);
  const reduceMotion = useReducedMotion();
  const reqIdRef = useRef(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const syncUrl = useCallback(
    (q: string, k: number, t: string, m: number) => {
      if (typeof window === "undefined") return;
      const params = new URLSearchParams();
      if (q) params.set("q", q);
      if (k !== TOP_K_DEFAULT) params.set("k", String(k));
      if (t) params.set("t", t);
      if (m > 0) params.set("m", String(m));
      const search = params.toString();
      const next = `${window.location.pathname}${search ? `?${search}` : ""}`;
      window.history.replaceState(null, "", next);
    },
    [],
  );

  const runRecall = useCallback(async () => {
    const trimmed = query.trim();
    if (!trimmed) return;
    const reqId = ++reqIdRef.current;
    setLoading(true);
    setError(null);
    syncUrl(trimmed, topK, memoryType, minImportance);
    try {
      const res = await fetch("/api/recall", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmed,
          top_k: topK,
          memory_type: memoryType || undefined,
          min_importance: minImportance > 0 ? minImportance : undefined,
        }),
      });
      const data = (await res.json().catch(() => ({}))) as {
        items?: RecallResult[];
        elapsed_ms?: number;
        error?: string;
      };
      if (!res.ok) {
        throw new Error(data.error ?? `HTTP ${res.status}`);
      }
      if (reqId !== reqIdRef.current) return;
      setItems(data.items ?? []);
      setElapsedMs(data.elapsed_ms ?? null);
      setHasRun(true);
    } catch (err) {
      if (reqId !== reqIdRef.current) return;
      setError(err instanceof Error ? err.message : String(err));
      setHasRun(true);
      setItems([]);
      setElapsedMs(null);
    } finally {
      if (reqId === reqIdRef.current) setLoading(false);
    }
  }, [query, topK, memoryType, minImportance, syncUrl]);

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    runRecall();
  };

  const onTextareaKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      runRecall();
    }
  };

  const terms = useMemo(() => tokenizeQuery(query), [query]);
  const hasQuery = query.trim().length > 0;

  return (
    <div className="space-y-5">
      <form onSubmit={onSubmit} className="space-y-3">
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onTextareaKey}
          placeholder="Ask the memory engine something… (Cmd/Ctrl+Enter to recall)"
          rows={3}
          spellCheck={false}
          className={cn(
            "w-full resize-y rounded-xl border border-border/60 bg-card/60",
            "px-3.5 py-2.5 text-sm text-foreground placeholder:text-muted-foreground/70",
            "outline-none transition-colors",
            "hover:border-border focus:border-foreground/40 focus:bg-card/80",
            "min-h-[64px]",
          )}
        />

        <div className="flex flex-wrap items-end gap-x-5 gap-y-3 text-xs">
          <label className="flex flex-col gap-1">
            <span className="text-muted-foreground">
              top_k <span className="tabular-nums text-foreground">{topK}</span>
            </span>
            <input
              type="range"
              min={TOP_K_MIN}
              max={TOP_K_MAX}
              step={1}
              value={topK}
              onChange={(e) => setTopK(Number.parseInt(e.target.value, 10))}
              className="w-40 accent-foreground"
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-muted-foreground">
              min_importance{" "}
              <span className="tabular-nums text-foreground">
                {minImportance}
              </span>
            </span>
            <input
              type="range"
              min={0}
              max={IMP_MAX}
              step={1}
              value={minImportance}
              onChange={(e) =>
                setMinImportance(Number.parseInt(e.target.value, 10))
              }
              className="w-40 accent-foreground"
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-muted-foreground">memory_type</span>
            <select
              value={memoryType}
              onChange={(e) => setMemoryType(e.target.value)}
              className={cn(
                "rounded-md border border-border/60 bg-card/60 px-2 py-1",
                "text-xs text-foreground outline-none transition-colors",
                "hover:border-border focus:border-foreground/40",
              )}
            >
              {TYPE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>

          <button
            type="submit"
            disabled={!hasQuery || loading}
            className={cn(
              "ml-auto inline-flex items-center gap-1.5 rounded-md border border-border/60",
              "bg-foreground px-3 py-1.5 text-xs font-medium text-background",
              "transition-colors hover:bg-foreground/90",
              "disabled:cursor-not-allowed disabled:opacity-50",
            )}
          >
            <Sparkles className="h-3 w-3" aria-hidden />
            {loading ? "recalling…" : "Recall"}
          </button>
        </div>
      </form>

      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
        <span>
          {loading
            ? "recalling… (graph + keyword + vector → rerank → MMR)"
            : hasRun
              ? `${items.length} result${items.length === 1 ? "" : "s"}${
                  elapsedMs !== null ? ` · ${elapsedMs}ms` : ""
                }`
              : "Type a query and press Recall to run the full recall pipeline (graph + keyword + vector → rerank → MMR)."}
        </span>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      {!hasRun ? null : items.length === 0 && !loading && !error ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full border border-dashed border-border/70 text-muted-foreground">
            <Sparkles className="h-5 w-5" aria-hidden />
          </div>
          <p className="text-sm text-foreground/90">No matches.</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Try a different phrasing or relax the filters.
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
                <div className="flex items-start gap-2">
                  <div
                    className={cn(
                      "mt-3.5 flex shrink-0 flex-col items-end gap-0.5",
                      "w-14 font-mono text-[10px] tabular-nums text-muted-foreground",
                    )}
                    title={`MMR rank #${i + 1} · score ${m.score.toFixed(3)}`}
                  >
                    <span className="text-foreground/80">#{i + 1}</span>
                    <span>{m.score.toFixed(2)}</span>
                  </div>
                  <div className="flex-1">
                    <MemoryCard
                      memory={m}
                      highlightTerms={terms}
                      dayBadge={localDayOf(m.created_at)}
                    />
                  </div>
                </div>
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </div>
  );
}
