"use client";

import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { Sun, Moon, Monitor } from "lucide-react";
import { Button } from "@/components/ui/button";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  function cycleTheme() {
    if (theme === "dark") setTheme("light");
    else if (theme === "light") setTheme("system");
    else setTheme("dark");
  }

  const Icon = !mounted
    ? Monitor
    : theme === "dark"
      ? Moon
      : theme === "light"
        ? Sun
        : Monitor;

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={cycleTheme}
      className="h-7 w-7 rounded-md border border-border/60 bg-card/60 text-muted-foreground hover:border-border hover:text-foreground"
      aria-label="Toggle theme"
      suppressHydrationWarning
    >
      <Icon className="h-3.5 w-3.5" aria-hidden suppressHydrationWarning />
    </Button>
  );
}
