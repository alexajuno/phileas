import "server-only";

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

function phileasHome(): string {
  return process.env.PHILEAS_HOME ?? join(homedir(), ".phileas");
}

export function daemonPort(): number | null {
  try {
    const raw = readFileSync(join(phileasHome(), "daemon.port"), "utf8").trim();
    const port = Number.parseInt(raw, 10);
    return Number.isFinite(port) && port > 0 ? port : null;
  } catch {
    return null;
  }
}

/**
 * Base URL for the daemon's JSON-RPC endpoint. `PHILEAS_API_URL` overrides the
 * local port-file discovery, letting the dashboard target a remote daemon (the
 * box) instead of `127.0.0.1`. Returns null when neither is available.
 */
export function daemonBaseUrl(): string | null {
  const override = process.env.PHILEAS_API_URL;
  if (override) return override.replace(/\/+$/, "");
  const port = daemonPort();
  return port === null ? null : `http://127.0.0.1:${port}`;
}

export class DaemonUnavailableError extends Error {
  constructor(message = "phileas daemon not running") {
    super(message);
    this.name = "DaemonUnavailableError";
  }
}

export class DaemonError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DaemonError";
  }
}

export async function callDaemon<T = unknown>(
  method: string,
  params: Record<string, unknown> = {},
): Promise<T> {
  const base = daemonBaseUrl();
  if (base === null) throw new DaemonUnavailableError();

  let res: Response;
  try {
    res = await fetch(base, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ method, params }),
      cache: "no-store",
    });
  } catch (err) {
    throw new DaemonUnavailableError(
      err instanceof Error ? err.message : String(err),
    );
  }

  const body = (await res.json().catch(() => ({}))) as {
    ok?: boolean;
    result?: T;
    error?: string;
  };

  if (!res.ok || body.ok === false) {
    throw new DaemonError(body.error ?? `daemon HTTP ${res.status}`);
  }
  return body.result as T;
}

/**
 * Map a daemon call error to an HTTP status for a route response:
 * daemon down -> 503, daemon-side error -> 502, anything else -> 500.
 */
export function daemonErrorStatus(err: unknown): number {
  if (err instanceof DaemonUnavailableError) return 503;
  if (err instanceof DaemonError) return 502;
  return 500;
}
