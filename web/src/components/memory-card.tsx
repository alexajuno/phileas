"use client";

import Link from "next/link";
import { useState } from "react";
import { ChevronDown } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { formatTime, toneFor } from "@/lib/format";
import { highlight } from "@/lib/highlight";
import type { MemoryItem } from "@/lib/types";
import { cn } from "@/lib/utils";

type Props = {
  memory: MemoryItem;
  justArrived?: boolean;
  highlightTerms?: readonly string[];
  dayBadge?: string;
};

export function MemoryCard({
  memory,
  justArrived,
  highlightTerms,
  dayBadge,
}: Props) {
  const [open, setOpen] = useState(false);
  const tone = toneFor(memory.memory_type);
  const hasRaw = Boolean(memory.raw_text);
  const terms = highlightTerms && highlightTerms.length > 0 ? highlightTerms : null;

  return (
    <article
      className={cn(
        "group relative rounded-xl border border-border/60 bg-card/60",
        "px-4 py-3.5 transition-all hover:border-border",
        "hover:translate-y-[-1px] hover:shadow-sm",
        justArrived && "ring-2 ring-offset-2 ring-offset-background " + tone.ring,
      )}
    >
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <time className="font-mono tabular-nums text-muted-foreground">
          {formatTime(memory.created_at)}
        </time>
        {dayBadge && (
          <Link
            href={`/?day=${dayBadge}`}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border border-border/60 bg-muted/40",
              "px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-muted-foreground",
              "transition-colors hover:border-border hover:text-foreground",
            )}
            title="Open this day"
          >
            {dayBadge}
          </Link>
        )}
        <span className={cn("inline-flex items-center gap-1.5", tone.text)}>
          <span className={cn("h-1.5 w-1.5 rounded-full", tone.dot)} />
          <span className="text-[11px] uppercase tracking-wide">
            {memory.memory_type}
          </span>
        </span>
        <span className="text-muted-foreground">
          imp <span className="text-foreground">{memory.importance}</span>
        </span>
        <span className="text-muted-foreground">tier {memory.tier}</span>
        {memory.reinforcement_count > 0 && (
          <span className="text-muted-foreground">
            ×{memory.reinforcement_count}
          </span>
        )}
        <span className="ml-auto font-mono text-[10px] text-muted-foreground/60">
          {memory.id.slice(0, 8)}
        </span>
      </div>

      <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-foreground/95">
        {terms ? highlight(memory.summary, terms) : memory.summary}
      </p>

      {(memory.tags.length > 0 || hasRaw) && (
        <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
          {memory.tags.map((t) => (
            <Badge
              key={t}
              variant="outline"
              className="rounded-full border-border/70 px-2 py-0 text-[10px] font-normal text-muted-foreground"
            >
              {terms ? highlight(t, terms) : t}
            </Badge>
          ))}
          {hasRaw && (
            <button
              type="button"
              onClick={() => setOpen((v) => !v)}
              className={cn(
                "ml-auto inline-flex items-center gap-1 rounded-md px-1.5 py-0.5",
                "text-[11px] text-muted-foreground transition-colors",
                "hover:text-foreground",
              )}
            >
              raw
              <ChevronDown
                className={cn(
                  "h-3 w-3 transition-transform",
                  open && "rotate-180",
                )}
              />
            </button>
          )}
        </div>
      )}

      {open && hasRaw && (
        <pre
          className={cn(
            "mt-3 max-h-64 overflow-auto rounded-lg border border-border/60",
            "bg-muted/40 p-3 font-mono text-[11.5px] leading-relaxed",
            "text-foreground/90 whitespace-pre-wrap",
          )}
        >
          {terms && memory.raw_text
            ? highlight(memory.raw_text, terms)
            : memory.raw_text}
        </pre>
      )}
    </article>
  );
}
