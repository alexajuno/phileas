"use client";

import { useEffect } from "react";
import { ChevronLeft, ChevronRight, CalendarDays } from "lucide-react";

import { Button, buttonVariants } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { formatDayLabel } from "@/lib/format";
import { shiftDay, todayLocal } from "@/lib/day";
import { cn } from "@/lib/utils";

type Props = {
  day: string;
  onChange: (day: string) => void;
};

export function DayNav({ day, onChange }: Props) {
  const today = todayLocal();
  const isToday = day === today;
  const isFuture = day > today;

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLElement) {
        const tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || e.target.isContentEditable) return;
      }
      if (e.key === "ArrowLeft") onChange(shiftDay(day, -1));
      else if (e.key === "ArrowRight" && !isToday) onChange(shiftDay(day, 1));
      else if (e.key.toLowerCase() === "t") onChange(today);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [day, isToday, today, onChange]);

  const [y, m, d] = day.split("-").map(Number);
  const selected = new Date(y, m - 1, d);

  return (
    <div className="flex items-center gap-1">
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        aria-label="Previous day"
        onClick={() => onChange(shiftDay(day, -1))}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>

      <Popover>
        <PopoverTrigger
          className={cn(
            buttonVariants({ variant: "ghost" }),
            "h-8 gap-2 px-2.5 font-normal text-foreground hover:bg-muted/60",
          )}
        >
          <CalendarDays className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="tabular-nums">{formatDayLabel(day)}</span>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align="start">
          <Calendar
            mode="single"
            selected={selected}
            onSelect={(date) => {
              if (!date) return;
              const p = (n: number) => String(n).padStart(2, "0");
              onChange(
                `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())}`,
              );
            }}
            disabled={{ after: new Date() }}
            autoFocus
          />
        </PopoverContent>
      </Popover>

      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        aria-label="Next day"
        disabled={isToday || isFuture}
        onClick={() => onChange(shiftDay(day, 1))}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>

      {!isToday && (
        <Button
          variant="outline"
          size="sm"
          className="ml-1 h-7 px-2.5 text-xs"
          onClick={() => onChange(today)}
        >
          Today
        </Button>
      )}
    </div>
  );
}
