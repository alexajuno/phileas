"use client";

import { useCallback, useEffect, useState } from "react";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import type {
  IngestionEventListItem,
  IngestionEventRow,
  IngestionHealth,
  LinkedMemoryRow,
} from "@/lib/phileas-db";

const POLL_MS = 30_000;

type Props = {
  initialHealth: IngestionHealth | null;
  initialEvents: IngestionEventListItem[];
  unavailable?: boolean;
};

type DetailPayload = {
  event: IngestionEventRow;
  memories: LinkedMemoryRow[];
};

function formatAbsTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function HealthCard({ health }: { health: IngestionHealth | null }) {
  if (!health) {
    return (
      <div className="rounded-lg border border-border/60 bg-card/40 px-4 py-6 text-sm text-muted-foreground">
        No health data.
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      <Tile
        label="Last hour"
        value={health.events_received_1h.toString()}
        sub="events captured"
      />
      <Tile
        label="Last 24h"
        value={health.events_received_24h.toString()}
        sub="events captured"
      />
      <Tile
        label="All-time"
        value={health.events_total.toString()}
        sub="events on file"
      />
    </div>
  );
}

function Tile({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-lg border border-border/60 bg-card/40 px-4 py-3">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 font-mono text-2xl tabular-nums text-foreground">
        {value}
      </div>
      {sub ? (
        <div className="mt-1 text-xs text-muted-foreground">{sub}</div>
      ) : null}
    </div>
  );
}

export function IngestionView({
  initialHealth,
  initialEvents,
  unavailable,
}: Props) {
  const [health, setHealth] = useState<IngestionHealth | null>(initialHealth);
  const [events, setEvents] = useState<IngestionEventListItem[]>(initialEvents);
  const [refreshing, setRefreshing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [openId, setOpenId] = useState<string | null>(null);
  const [detail, setDetail] = useState<DetailPayload | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setRefreshing(true);
    try {
      const url = new URL(
        "/api/monitoring/ingestion",
        window.location.origin,
      );
      url.searchParams.set("limit", "50");
      const res = await fetch(url.toString(), { cache: "no-store" });
      if (!res.ok) return;
      const body = (await res.json()) as {
        health: IngestionHealth;
        events: IngestionEventListItem[];
      };
      setHealth(body.health);
      setEvents(body.events);
    } catch {
      // swallow — keep last known state
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    if (!autoRefresh || unavailable) return;
    const id = window.setInterval(() => {
      fetchData();
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [autoRefresh, unavailable, fetchData]);

  const openDetail = useCallback(async (id: string) => {
    setOpenId(id);
    setDetail(null);
    setDetailError(null);
    setDetailLoading(true);
    try {
      const res = await fetch(
        `/api/monitoring/ingestion/event/${encodeURIComponent(id)}`,
        { cache: "no-store" },
      );
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as {
          error?: string;
        };
        setDetailError(body.error ?? `HTTP ${res.status}`);
        return;
      }
      const body = (await res.json()) as DetailPayload;
      setDetail(body);
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : String(err));
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const closeDetail = useCallback(() => {
    setOpenId(null);
    setDetail(null);
    setDetailError(null);
  }, []);

  if (unavailable) {
    return (
      <div className="rounded-lg border border-border/60 bg-card/40 px-4 py-6 text-sm text-muted-foreground">
        Phileas memory database not available. Start the daemon, or set
        <code className="mx-1 rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
          PHILEAS_HOME
        </code>
        to point at one.
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <HealthCard health={health} />

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-muted-foreground">{events.length} events</span>
        <span className="ml-auto flex items-center gap-2">
          <label className="flex items-center gap-1 text-muted-foreground">
            <input
              type="checkbox"
              className="h-3 w-3 accent-foreground"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            auto-refresh 30s
          </label>
          <button
            type="button"
            onClick={() => fetchData()}
            className="rounded border border-border/60 bg-card/60 px-2 py-1 text-foreground hover:border-border"
          >
            {refreshing ? "refreshing…" : "refresh"}
          </button>
        </span>
      </div>

      <div className="overflow-hidden rounded-lg border border-border/60">
        <table className="w-full text-sm">
          <thead className="bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Received</th>
              <th className="px-3 py-2 text-left font-medium">Preview</th>
            </tr>
          </thead>
          <tbody>
            {events.length === 0 ? (
              <tr>
                <td
                  colSpan={2}
                  className="px-3 py-8 text-center text-sm text-muted-foreground"
                >
                  No events yet.
                </td>
              </tr>
            ) : (
              events.map((ev) => (
                <tr
                  key={ev.id}
                  onClick={() => openDetail(ev.id)}
                  className="cursor-pointer border-t border-border/40 hover:bg-muted/20"
                >
                  <td className="whitespace-nowrap px-3 py-2 font-mono text-xs text-muted-foreground">
                    {formatAbsTime(ev.received_at)}
                  </td>
                  <td className="max-w-0 px-3 py-2">
                    <div className="truncate text-xs text-muted-foreground">
                      {ev.text_preview}
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <Dialog
        open={openId != null}
        onOpenChange={(open) => {
          if (!open) closeDetail();
        }}
      >
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>Event detail</DialogTitle>
          </DialogHeader>
          {detailLoading ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : detailError ? (
            <p className="text-sm text-red-400">{detailError}</p>
          ) : detail ? (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                <span className="font-mono">{detail.event.id}</span>
                <span>·</span>
                <span>{formatAbsTime(detail.event.received_at)}</span>
              </div>
              <div>
                <div className="mb-1 text-[11px] uppercase tracking-wide text-muted-foreground">
                  Event text
                </div>
                <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded border border-border/60 bg-card/40 px-3 py-2 font-mono text-xs">
                  {detail.event.text}
                </pre>
              </div>
              <div>
                <div className="mb-1 flex items-center gap-2 text-[11px] uppercase tracking-wide text-muted-foreground">
                  <span>Linked memories</span>
                  <span
                    className={cn(
                      "font-mono normal-case",
                      detail.memories.length === 0 && "text-muted-foreground",
                    )}
                  >
                    {detail.memories.length}
                  </span>
                </div>
                {detail.memories.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    Nothing memory-worthy was extracted from this event.
                    Either Claude judged the turn not worth saving, or the
                    daemon was unreachable when the hint fired (in which
                    case the link wasn&apos;t set).
                  </p>
                ) : (
                  <ul className="space-y-2">
                    {detail.memories.map((m) => (
                      <li
                        key={m.id}
                        className="rounded border border-border/60 bg-card/40 px-3 py-2"
                      >
                        <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                          <span className="rounded bg-muted px-1.5 py-0.5 font-mono">
                            {m.memory_type}
                          </span>
                          <span className="ml-auto font-mono">
                            {formatAbsTime(m.created_at)}
                          </span>
                        </div>
                        <p className="mt-1 text-sm">{m.summary}</p>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}
