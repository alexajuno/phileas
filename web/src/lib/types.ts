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
  tags: string[];
  daily_ref: string | null;
  source_session_id: string | null;
  created_at: string;
  updated_at: string;
};

export type DayCount = {
  day: string;
  count: number;
};
