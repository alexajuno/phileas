import type { MemoryType } from "./types";

export function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function formatDayLabel(day: string): string {
  const [y, m, d] = day.split("-").map(Number);
  const dt = new Date(y, m - 1, d);
  return dt.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export const TYPE_TONE: Record<MemoryType | "default", {
  ring: string; dot: string; text: string;
}> = {
  event:       { ring: "ring-emerald-500/30", dot: "bg-emerald-400", text: "text-emerald-300" },
  knowledge:   { ring: "ring-sky-500/30",     dot: "bg-sky-400",     text: "text-sky-300" },
  reflection:  { ring: "ring-violet-500/30",  dot: "bg-violet-400",  text: "text-violet-300" },
  behavior:    { ring: "ring-amber-500/30",   dot: "bg-amber-400",   text: "text-amber-300" },
  profile:     { ring: "ring-rose-500/30",    dot: "bg-rose-400",    text: "text-rose-300" },
  feedback:    { ring: "ring-orange-500/30",  dot: "bg-orange-400",  text: "text-orange-300" },
  observation: { ring: "ring-teal-500/30",    dot: "bg-teal-400",    text: "text-teal-300" },
  preference:  { ring: "ring-fuchsia-500/30", dot: "bg-fuchsia-400", text: "text-fuchsia-300" },
  project:     { ring: "ring-indigo-500/30",  dot: "bg-indigo-400",  text: "text-indigo-300" },
  reference:   { ring: "ring-slate-500/30",   dot: "bg-slate-400",   text: "text-slate-300" },
  default:     { ring: "ring-neutral-500/20", dot: "bg-neutral-400", text: "text-neutral-300" },
};

export function toneFor(type: string) {
  return TYPE_TONE[type as MemoryType] ?? TYPE_TONE.default;
}
