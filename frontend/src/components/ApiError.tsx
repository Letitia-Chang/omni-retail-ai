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
        Tried <code className="font-mono text-xs">{API_BASE_URL}</code>. Make sure the FastAPI
        backend is running and CORS allows this origin.
      </p>
      <p className="mt-2 text-xs text-muted-foreground">Details: {message}</p>
    </div>
  );
}

export function Loading({ rows = 4 }: { rows?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-16 rounded-lg bg-muted animate-pulse" />
      ))}
    </div>
  );
}
