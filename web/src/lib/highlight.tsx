import type { ReactNode } from "react";

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function highlight(
  text: string,
  terms: readonly string[],
): ReactNode[] {
  const cleaned = terms.map((t) => t.trim()).filter(Boolean);
  if (cleaned.length === 0) return [text];

  const pattern = new RegExp(
    `(${cleaned.map(escapeRegex).join("|")})`,
    "gi",
  );
  const parts = text.split(pattern);
  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <mark
        key={i}
        className="rounded-sm bg-amber-300/20 px-0.5 text-amber-100"
      >
        {part}
      </mark>
    ) : (
      part
    ),
  );
}

export function tokenizeQuery(q: string): string[] {
  return q
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 8);
}
