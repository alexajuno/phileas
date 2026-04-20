export function isValidDay(day: string): boolean {
  return /^\d{4}-\d{2}-\d{2}$/.test(day) && !Number.isNaN(Date.parse(day));
}

export function localDayBoundsAsUtcIso(day: string): { startIso: string; endIso: string } {
  const [y, m, d] = day.split("-").map(Number);
  const start = new Date(y, m - 1, d, 0, 0, 0, 0);
  const end = new Date(y, m - 1, d + 1, 0, 0, 0, 0);
  return { startIso: start.toISOString(), endIso: end.toISOString() };
}

export function todayLocal(): string {
  const now = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())}`;
}

export function shiftDay(day: string, delta: number): string {
  const [y, m, d] = day.split("-").map(Number);
  const dt = new Date(y, m - 1, d + delta);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${dt.getFullYear()}-${p(dt.getMonth() + 1)}-${p(dt.getDate())}`;
}
