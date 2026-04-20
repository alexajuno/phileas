import { toneFor } from "@/lib/format";
import type { MemoryItem } from "@/lib/types";
import { cn } from "@/lib/utils";

export function StatsStrip({ items }: { items: MemoryItem[] }) {
  const counts = items.reduce<Record<string, number>>((acc, m) => {
    acc[m.memory_type] = (acc[m.memory_type] ?? 0) + 1;
    return acc;
  }, {});
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="inline-flex items-baseline gap-1.5 rounded-full border border-border/70 bg-muted/40 px-3 py-1">
        <span className="font-mono text-sm font-medium tabular-nums">
          {items.length}
        </span>
        <span className="text-muted-foreground">memories</span>
      </span>
      {entries.map(([type, n]) => {
        const tone = toneFor(type);
        return (
          <span
            key={type}
            className="inline-flex items-center gap-1.5 rounded-full border border-border/60 px-2.5 py-1 text-muted-foreground"
          >
            <span className={cn("h-1.5 w-1.5 rounded-full", tone.dot)} />
            <span className={tone.text}>{type}</span>
            <span className="font-mono tabular-nums text-foreground">{n}</span>
          </span>
        );
      })}
    </div>
  );
}
