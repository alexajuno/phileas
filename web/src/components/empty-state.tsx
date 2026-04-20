export function EmptyState({ day, isToday }: { day: string; isToday: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full border border-dashed border-border/70 text-muted-foreground">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-5 w-5"
          aria-hidden
        >
          <path d="M4 7a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2Z" />
          <path d="M4 11h16" />
          <path d="M9 3v4" />
          <path d="M15 3v4" />
        </svg>
      </div>
      <p className="text-sm text-foreground/90">
        {isToday ? "Nothing captured yet today." : "No memories on this day."}
      </p>
      <p className="mt-1 text-xs text-muted-foreground">
        {isToday
          ? "New memories will appear here as Phileas ingests your sessions."
          : day}
      </p>
    </div>
  );
}
