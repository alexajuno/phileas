export type MemoryType =
  | "profile"
  | "event"
  | "knowledge"
  | "behavior"
  | "reflection"
  | "feedback"
  | "observation"
  | "preference"
  | "project"
  | "reference";

export type MemoryItem = {
  id: string;
  summary: string;
  memory_type: MemoryType | string;
  importance: number;
  status: string;
  access_count: number;
  reinforcement_count: number;
  last_reinforced: string | null;
  raw_text: string | null;
  daily_ref: string | null;
  created_at: string;
  updated_at: string;
};

export type DayCount = {
  day: string;
  count: number;
};

export type EntitySummary = {
  name: string;
  type: string;
  types: string[];
  aliases: string[];
  memory_count: number;
};

export type EntityRelation = {
  name: string;
  type: string;
  types?: string[];
  edge_type: string;
  direction: "out" | "in";
};

export type EntityDetail = {
  name: string;
  type: string;
  types: string[];
  aliases: string[];
  description: string;
  props: Record<string, unknown>;
  relations: EntityRelation[];
  memories: MemoryItem[];
};

// --- Monitoring: recall traces (served by the daemon, mirrors stats/queries.py) ---

export type TraceSource =
  | "hook_dispatch"
  | "engine.recall"
  | "engine.recall_raw"
  | "engine.recall_recent";

export interface TraceRow {
  id: number;
  created_at: string;
  source: TraceSource | string;
  query: string | null;
  latency_ms: number | null;
  candidate_count: number | null;
  returned_ids: string[] | null;
  pool_chars: number | null;
  extra: Record<string, unknown> | null;
}

export interface AggregateRow {
  source: string;
  count: number;
  p50: number | null;
  p90: number | null;
  p99: number | null;
  avg_pool_chars: number | null;
  avg_candidates: number | null;
}

export interface AggregateResult {
  since: string;
  by_source: AggregateRow[];
  total: number;
}

export interface BucketStats {
  count: number;
  candidate_p50: number | null;
  candidate_p95: number | null;
  latency_p50: number | null;
  latency_p95: number | null;
  source_mix: Record<string, number>;
  hop_distribution: Record<string, number>;
}

export interface CompareResult {
  cutoff: string;
  source: string;
  since: string;
  before: BucketStats;
  after: BucketStats;
}

// --- Monitoring: ingestion (events table) ---

export type IngestionEventRow = {
  id: string;
  text: string;
  received_at: string;
};

export type IngestionEventListItem = Omit<IngestionEventRow, "text"> & {
  text_preview: string;
};

export type IngestionHealth = {
  events_received_1h: number;
  events_received_24h: number;
  events_total: number;
};

export type LinkedMemoryRow = {
  id: string;
  summary: string;
  memory_type: string;
  importance: number;
  created_at: string;
};
