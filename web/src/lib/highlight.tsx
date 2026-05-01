import type { ReactNode } from "react";

// Short function/pronoun words that are noise as highlight terms. Mirrors the
// stop-word list used by the engine's recall keyword path so the highlight
// reflects the same notion of "meaningful query word".
const STOP_WORDS = new Set([
  "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
  "of", "with", "by", "from", "is", "it", "its", "be", "as", "that",
  "this", "was", "are", "were", "been", "have", "has", "had", "do", "did",
  "does", "will", "would", "could", "should", "may", "might", "shall",
  "can", "not", "no", "so", "if", "then", "than", "about", "us", "we",
  "i", "you", "he", "she", "they", "me", "him", "her", "them", "my",
  "our", "your", "his", "their", "still", "just", "also", "up", "out",
  "what", "which", "who", "when", "where", "how", "why",
]);

const MAX_TERMS = 8;

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function highlight(
  text: string,
  terms: readonly string[],
): ReactNode[] {
  const cleaned = terms.map((t) => t.trim()).filter(Boolean);
  if (cleaned.length === 0) return [text];

  // Word-boundary match so "ex" doesn't paint inside "explicitly". Apostrophes
  // are non-word characters in JS regex, so \bgiao\b still hits "Giao" inside
  // "Giao's" — exactly what we want.
  const pattern = new RegExp(
    `\\b(${cleaned.map(escapeRegex).join("|")})\\b`,
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
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of q.trim().split(/\s+/)) {
    // Strip leading/trailing punctuation; drop possessive 's so "Giao's"
    // tokenizes to "Giao". Internal apostrophes/hyphens are kept (e.g. "won't").
    const stripped = raw
      .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "")
      .replace(/['’]s$/i, "");
    if (stripped.length < 2) continue;
    const key = stripped.toLowerCase();
    if (STOP_WORDS.has(key)) continue;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(stripped);
    if (out.length >= MAX_TERMS) break;
  }
  return out;
}
