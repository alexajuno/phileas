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
  const port = daemonPort();
  if (port === null) throw new DaemonUnavailableError();

  let res: Response;
  try {
    res = await fetch(`http://127.0.0.1:${port}`, {
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
