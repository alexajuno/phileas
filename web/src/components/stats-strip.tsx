import { toneFor } from "@/lib/format";
import type { MemoryItem } from "@/lib/types";
import { cn } from "@/lib/utils";

type Props = {
  items: MemoryItem[];
  selectedType: string | null;
  onSelect: (type: string | null) => void;
};

export function StatsStrip({ items, selectedType, onSelect }: Props) {
  const counts = items.reduce<Record<string, number>>((acc, m) => {
    acc[m.memory_type] = (acc[m.memory_type] ?? 0) + 1;
    return acc;
  }, {});
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const allActive = selectedType === null;

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <button
        type="button"
        onClick={() => onSelect(null)}
        aria-pressed={allActive}
        className={cn(
          "inline-flex items-baseline gap-1.5 rounded-full border px-3 py-1 transition",
          allActive
            ? "border-foreground/40 bg-muted/60 ring-1 ring-foreground/20"
            : "border-border/70 bg-muted/40 hover:bg-muted/60",
        )}
      >
        <span className="font-mono text-sm font-medium tabular-nums">
          {items.length}
        </span>
        <span className="text-muted-foreground">
          {allActive ? "memories" : "all"}
        </span>
      </button>
      {entries.map(([type, n]) => {
        const tone = toneFor(type);
        const active = selectedType === type;
        return (
          <button
            key={type}
            type="button"
            onClick={() => onSelect(active ? null : type)}
            aria-pressed={active}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-muted-foreground transition",
              active
                ? cn("border-transparent ring-1", tone.ring, "bg-muted/60")
                : "border-border/60 hover:bg-muted/40",
            )}
          >
            <span className={cn("h-1.5 w-1.5 rounded-full", tone.dot)} />
            <span className={tone.text}>{type}</span>
            <span className="font-mono tabular-nums text-foreground">{n}</span>
          </button>
        );
      })}
    </div>
  );
}
