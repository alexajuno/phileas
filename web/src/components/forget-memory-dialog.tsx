"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { MemoryItem } from "@/lib/types";

type Props = {
  memory: MemoryItem;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onForgotten: (id: string) => void;
};

export function ForgetMemoryDialog({
  memory,
  open,
  onOpenChange,
  onForgotten,
}: Props) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleConfirm() {
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`/api/memories/${encodeURIComponent(memory.id)}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? `HTTP ${res.status}`);
      }
      onForgotten(memory.id);
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (submitting) return;
        if (!next) setError(null);
        onOpenChange(next);
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Forget this memory?</DialogTitle>
          <DialogDescription>
            Archives the memory and removes its embeddings. The DB row is kept
            with <code className="font-mono">status=&apos;archived&apos;</code>.
          </DialogDescription>
        </DialogHeader>

        <div className="rounded-lg border border-border/60 bg-muted/40 p-3">
          <div className="mb-1.5 flex items-center gap-2 text-[11px] uppercase tracking-wide text-muted-foreground">
            <span>{memory.memory_type}</span>
            <span>·</span>
            <span>imp {memory.importance}</span>
            <span>·</span>
            <span className="font-mono normal-case">
              {memory.id.slice(0, 8)}
            </span>
          </div>
          <p className="line-clamp-3 text-sm text-foreground/90">
            {memory.summary}
          </p>
        </div>

        {error && (
          <p className="text-xs text-destructive" role="alert">
            {error}
          </p>
        )}

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={submitting}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleConfirm}
            disabled={submitting}
          >
            {submitting ? "Forgetting…" : "Forget"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
