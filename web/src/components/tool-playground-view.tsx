"use client";

import { useCallback, useMemo, useState, type FormEvent } from "react";
import { Play } from "lucide-react";

import { MemoryCard } from "./memory-card";
import { localDayOf } from "@/lib/day";
import type { MemoryItem } from "@/lib/types";
import { cn } from "@/lib/utils";

type FieldType = "int" | "text" | "bool";

type Field = {
  key: string;
  label: string;
  type: FieldType;
  default?: string | number | boolean;
  required?: boolean;
  placeholder?: string;
};

type ToolSpec = {
  name: string;
  label: string;
  hint: string;
  fields: Field[];
};

// Mirrors the read-only recall-family signatures in src/phileas/tool_runner.py.
const TOOLS: ToolSpec[] = [
  {
    name: "recall_recent",
    label: "recall_recent",
    hint: "Top memories per day for the last N days — time-relative queries.",
    fields: [
      { key: "days", label: "days", type: "int", default: 7 },
      { key: "top_per_day", label: "top_per_day", type: "int", default: 10 },
    ],
  },
  {
    name: "timeline",
    label: "timeline",
    hint: "Memories anchored to a date or date range.",
    fields: [
      { key: "start_date", label: "start_date", type: "text", required: true, placeholder: "YYYY-MM-DD" },
      { key: "end_date", label: "end_date", type: "text", placeholder: "YYYY-MM-DD (optional)" },
      { key: "window", label: "window", type: "int", default: 1 },
    ],
  },
  {
    name: "about",
    label: "about",
    hint: "Memories connected to an entity in the knowledge graph.",
    fields: [
      { key: "name", label: "name", type: "text", required: true, placeholder: "entity name" },
      { key: "entity_type", label: "entity_type", type: "text", placeholder: "Person, Technology… (optional)" },
      { key: "memory_type", label: "memory_type", type: "text", placeholder: "profile, behavior… (optional)" },
      { key: "expand", label: "expand (1-hop fanout)", type: "bool", default: false },
    ],
  },
  {
    name: "list_day_memories",
    label: "list_day_memories",
    hint: "Every active memory anchored to one day (no window).",
    fields: [{ key: "date", label: "date", type: "text", placeholder: "YYYY-MM-DD (default today)" }],
  },
  {
    name: "serendipity",
    label: "serendipity",
    hint: "N high-signal memories NOT gated on query relevance.",
    fields: [
      { key: "n", label: "n", type: "int", default: 3 },
      { key: "exclude_ids", label: "exclude_ids", type: "text", placeholder: "id8a, id8b (comma-separated, optional)" },
    ],
  },
  {
    name: "hydrate",
    label: "hydrate",
    hint: "Full record of one memory — the drill-in for a pointer.",
    fields: [{ key: "memory_id", label: "memory_id", type: "text", required: true, placeholder: "id or id8 prefix" }],
  },
  {
    name: "thread",
    label: "thread",
    hint: "Verbatim source event + every memory extracted from it.",
    fields: [{ key: "event_id", label: "event_id", type: "text", required: true, placeholder: "event UUID" }],
  },
  {
    name: "find_entities",
    label: "find_entities",
    hint: "Candidate entities whose name/alias contains the query.",
    fields: [{ key: "query", label: "query", type: "text", required: true, placeholder: "name fragment, e.g. huyen" }],
  },
  {
    name: "scopes",
    label: "scopes",
    hint: "SCOPED_TO contexts of one memory — none means globally valid.",
    fields: [{ key: "memory_id", label: "memory_id", type: "text", required: true, placeholder: "id or id8 prefix" }],
  },
];

type Values = Record<string, string | boolean>;
type ToolResult = {
  text: string;
  cards: MemoryItem[];
  count: number;
  tokens: number;
  elapsed_ms: number;
};

function initialValues(spec: ToolSpec): Values {
  const v: Values = {};
  for (const f of spec.fields) {
    if (f.type === "bool") v[f.key] = f.default === true;
    else v[f.key] = f.default === undefined ? "" : String(f.default);
  }
  return v;
}

function buildArgs(spec: ToolSpec, values: Values): Record<string, unknown> {
  const args: Record<string, unknown> = {};
  for (const f of spec.fields) {
    const v = values[f.key];
    if (f.type === "bool") {
      if (v === true) args[f.key] = true; // omit the default-false case
      continue;
    }
    const s = String(v ?? "").trim();
    if (s === "") continue;
    if (f.type === "int") {
      const n = Number.parseInt(s, 10);
      if (Number.isFinite(n)) args[f.key] = n;
      continue;
    }
    if (f.key === "exclude_ids") {
      const arr = s.split(",").map((x) => x.trim()).filter(Boolean);
      if (arr.length > 0) args[f.key] = arr;
      continue;
    }
    args[f.key] = s;
  }
  return args;
}

export function ToolPlaygroundView() {
  const [toolName, setToolName] = useState(TOOLS[0].name);
  const spec = useMemo(
    () => TOOLS.find((t) => t.name === toolName) ?? TOOLS[0],
    [toolName],
  );
  const [values, setValues] = useState<Values>(() => initialValues(TOOLS[0]));
  const [result, setResult] = useState<ToolResult | null>(null);
  const [mode, setMode] = useState<"text" | "cards">("text");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onToolChange = useCallback((name: string) => {
    const next = TOOLS.find((t) => t.name === name) ?? TOOLS[0];
    setToolName(next.name);
    setValues(initialValues(next));
    setResult(null);
    setError(null);
  }, []);

  const setField = useCallback((key: string, val: string | boolean) => {
    setValues((prev) => ({ ...prev, [key]: val }));
  }, []);

  const run = useCallback(async () => {
    const missing = spec.fields.find(
      (f) => f.required && String(values[f.key] ?? "").trim() === "",
    );
    if (missing) {
      setError(`${missing.key} is required`);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/tool", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tool: spec.name, args: buildArgs(spec, values) }),
      });
      const data = (await res.json().catch(() => ({}))) as Partial<ToolResult> & {
        error?: string;
      };
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      const next: ToolResult = {
        text: data.text ?? "",
        cards: data.cards ?? [],
        count: data.count ?? 0,
        tokens: data.tokens ?? 0,
        elapsed_ms: data.elapsed_ms ?? 0,
      };
      setResult(next);
      setMode(next.cards.length > 0 ? "cards" : "text");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [spec, values]);

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    run();
  };

  return (
    <div className="space-y-5">
      <form onSubmit={onSubmit} className="space-y-4">
        <div className="flex flex-col gap-1">
          <span className="text-xs text-muted-foreground">tool</span>
          <select
            value={toolName}
            onChange={(e) => onToolChange(e.target.value)}
            className={cn(
              "w-full max-w-xs rounded-md border border-border/60 bg-card/60 px-2.5 py-1.5",
              "text-sm text-foreground outline-none transition-colors",
              "hover:border-border focus:border-foreground/40",
            )}
          >
            {TOOLS.map((t) => (
              <option key={t.name} value={t.name}>
                {t.label}
              </option>
            ))}
          </select>
          <span className="text-[11px] text-muted-foreground/80">{spec.hint}</span>
        </div>

        <div className="flex flex-wrap items-end gap-x-5 gap-y-3 text-xs">
          {spec.fields.map((f) =>
            f.type === "bool" ? (
              <label key={f.key} className="flex items-center gap-2 pb-1.5">
                <input
                  type="checkbox"
                  checked={values[f.key] === true}
                  onChange={(e) => setField(f.key, e.target.checked)}
                  className="accent-foreground"
                />
                <span className="text-muted-foreground">{f.label}</span>
              </label>
            ) : (
              <label key={f.key} className="flex flex-col gap-1">
                <span className="text-muted-foreground">
                  {f.label}
                  {f.required && <span className="text-destructive"> *</span>}
                </span>
                <input
                  type={f.type === "int" ? "number" : "text"}
                  value={String(values[f.key] ?? "")}
                  placeholder={f.placeholder}
                  onChange={(e) => setField(f.key, e.target.value)}
                  spellCheck={false}
                  className={cn(
                    "rounded-md border border-border/60 bg-card/60 px-2 py-1",
                    "text-xs text-foreground outline-none transition-colors",
                    "placeholder:text-muted-foreground/60",
                    "hover:border-border focus:border-foreground/40",
                    f.type === "int" ? "w-24" : "w-56",
                  )}
                />
              </label>
            ),
          )}

          <button
            type="submit"
            disabled={loading}
            className={cn(
              "ml-auto inline-flex items-center gap-1.5 rounded-md border border-border/60",
              "bg-foreground px-3 py-1.5 text-xs font-medium text-background",
              "transition-colors hover:bg-foreground/90",
              "disabled:cursor-not-allowed disabled:opacity-50",
            )}
          >
            <Play className="h-3 w-3" aria-hidden />
            {loading ? "running…" : "Run"}
          </button>
        </div>
      </form>

      <div className="flex items-center justify-between gap-3 text-[11px] text-muted-foreground">
        <span>
          {loading
            ? `running ${spec.name}…`
            : result
              ? `${result.count} item${result.count === 1 ? "" : "s"} · ~${result.tokens.toLocaleString()} tok · ${result.elapsed_ms}ms`
              : "Pick a tool, fill the args, and Run to see exactly what the model receives."}
        </span>
        {result && (
          <div className="inline-flex overflow-hidden rounded-md border border-border/60">
            {(["text", "cards"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={cn(
                  "px-2.5 py-1 text-[11px] transition-colors",
                  mode === m
                    ? "bg-foreground text-background"
                    : "bg-card/60 text-muted-foreground hover:text-foreground",
                )}
              >
                {m === "text" ? "Model output" : `Cards (${result.cards.length})`}
              </button>
            ))}
          </div>
        )}
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      {result && mode === "text" && (
        <pre
          className={cn(
            "max-h-[60vh] overflow-auto rounded-lg border border-border/60",
            "bg-muted/40 p-3 font-mono text-[11.5px] leading-relaxed",
            "whitespace-pre-wrap text-foreground/90",
          )}
        >
          {result.text || "(empty)"}
        </pre>
      )}

      {result && mode === "cards" &&
        (result.cards.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border/70 px-3 py-6 text-center text-xs text-muted-foreground">
            No memory cards for this tool — switch to Model output.
          </div>
        ) : (
          <ul className="space-y-2.5">
            {result.cards.map((m) => (
              <li key={m.id}>
                <MemoryCard memory={m} dayBadge={localDayOf(m.created_at)} />
              </li>
            ))}
          </ul>
        ))}
    </div>
  );
}
