import "server-only";

import { homedir } from "node:os";
import { join } from "node:path";
import Database, { type Database as DB } from "better-sqlite3";

let cached: DB | null = null;
let cachedPath: string | null = null;

function resolveDbPath(): string {
  const home = process.env.PHILEAS_HOME ?? join(homedir(), ".phileas");
  return join(home, "memory.db");
}

export function getDb(): DB {
  const path = resolveDbPath();
  if (cached && cachedPath === path) return cached;
  if (cached) {
    cached.close();
    cached = null;
  }
  const db = new Database(path, { readonly: true, fileMustExist: true });
  db.pragma("query_only = ON");
  cached = db;
  cachedPath = path;
  return db;
}

export function dbPath(): string {
  return resolveDbPath();
}
