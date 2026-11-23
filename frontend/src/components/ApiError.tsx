import { useEffect, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { API_BASE_URL } from "@/lib/api";

export function ApiError({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-5 text-sm">
      <div className="flex items-center gap-2 font-medium text-destructive">
        <AlertTriangle className="h-4 w-4" />
        Couldn't reach the OmniRetail API
      </div>
      <p className="mt-2 text-muted-foreground">
        The demo server didn't respond at{" "}
        <code className="font-mono text-xs">{API_BASE_URL}</code>. It runs on a free tier and may
        still be waking up. Please refresh the page in a minute.
      </p>
      <p className="mt-2 text-xs text-muted-foreground">Details: {message}</p>
    </div>
  );
}

export function Loading({ rows = 4 }: { rows?: number }) {
  // After a few seconds, explain the cold start so visitors don't assume the
  // demo is broken while the free-tier backend wakes up.
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setSlow(true), 4000);
    return () => clearTimeout(t);
  }, []);
  return (
    <div className="space-y-3">
      {slow && (
        <p className="text-sm text-muted-foreground">
          Waking up the demo server. This can take about a minute on the first visit.
        </p>
      )}
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-16 rounded-lg bg-muted animate-pulse" />
      ))}
    </div>
  );
}
