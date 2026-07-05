import { colorForSegment } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ArrowUpRight, Users } from "lucide-react";

export function SegmentCard({
  name,
  count,
  avgScore,
  active,
  onClick,
}: {
  name: string;
  count: number;
  avgScore: number;
  active: boolean;
  onClick: () => void;
}) {
  const color = colorForSegment(name);
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "group relative text-left rounded-xl border bg-card p-4 transition-all overflow-hidden",
        "hover:-translate-y-0.5 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-ring",
        active
          ? "border-primary shadow-md ring-1 ring-primary/30"
          : "border-border hover:border-primary/40"
      )}
    >
      {/* Color rail */}
      <div
        className="absolute inset-x-0 top-0 h-1"
        style={{ background: color }}
        aria-hidden
      />
      <div className="flex items-start justify-between gap-3">
        <div
          className="flex h-9 w-9 items-center justify-center rounded-lg"
          style={{
            background: `color-mix(in oklab, ${color} 18%, var(--card))`,
            color,
          }}
        >
          <Users className="h-4 w-4" />
        </div>
        <ArrowUpRight
          className={cn(
            "h-4 w-4 transition-all",
            active
              ? "text-primary translate-x-0.5 -translate-y-0.5"
              : "text-muted-foreground group-hover:text-foreground"
          )}
        />
      </div>
      <div className="mt-3 font-semibold text-sm leading-snug line-clamp-2">{name}</div>
      <div className="mt-2 flex items-center gap-3 text-xs text-muted-foreground">
        <span className="tabular-nums">
          <span className="font-semibold text-foreground">{count}</span> recs
        </span>
        <span className="tabular-nums">
          avg <span className="font-semibold text-foreground">{avgScore.toFixed(0)}%</span>
        </span>
      </div>
    </button>
  );
}
