import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import {
  api,
  explanationOf,
  productOf,
  scoreOf,
  strategyOf,
} from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { ApiError, Loading } from "@/components/ApiError";
import { CampaignCard } from "@/components/CampaignCard";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Search, SlidersHorizontal } from "lucide-react";

export const Route = createFileRoute("/campaigns")({
  component: CampaignsPage,
});

type SortKey = "score" | "product" | "segment" | "strategy";

function CampaignsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["campaigns"],
    queryFn: api.campaigns,
  });

  const [q, setQ] = useState("");
  const [strategy, setStrategy] = useState<string>("all");
  const [segment, setSegment] = useState<string>("all");
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [page, setPage] = useState(1);
  const pageSize = 12;

  const strategies = useMemo(() => {
    const s = new Set<string>();
    (data ?? []).forEach((c) => s.add(strategyOf(c)));
    return Array.from(s).sort();
  }, [data]);

  const segments = useMemo(() => {
    const s = new Set<string>();
    (data ?? []).forEach((c) => {
      if (c.customer_segment) s.add(c.customer_segment as string);
    });
    return Array.from(s).sort();
  }, [data]);

  const filtered = useMemo(() => {
    let rows = data ?? [];
    if (strategy !== "all") rows = rows.filter((c) => strategyOf(c) === strategy);
    if (segment !== "all") rows = rows.filter((c) => c.customer_segment === segment);
    if (q.trim()) {
      const term = q.toLowerCase();
      rows = rows.filter((c) =>
        [productOf(c), c.customer_segment, strategyOf(c), explanationOf(c)]
          .filter(Boolean)
          .some((v) => String(v).toLowerCase().includes(term))
      );
    }
    rows = [...rows].sort((a, b) => {
      switch (sortKey) {
        case "score":
          return scoreOf(b) - scoreOf(a);
        case "product":
          return productOf(a).localeCompare(productOf(b));
        case "segment":
          return String(a.customer_segment ?? "").localeCompare(
            String(b.customer_segment ?? "")
          );
        case "strategy":
          return strategyOf(a).localeCompare(strategyOf(b));
      }
    });
    return rows;
  }, [data, q, strategy, segment, sortKey]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const slice = filtered.slice((safePage - 1) * pageSize, safePage * pageSize);

  return (
    <>
      <PageHeader
        title="All campaign recommendations"
        description="Browse the full library of AI-ranked picks. Filter by segment, strategy, or search by product."
      />

      {error && <ApiError error={error} />}

      <Card className="mb-5">
        <CardContent className="p-4 md:p-5">
          <div className="flex flex-col md:flex-row gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                value={q}
                onChange={(e) => {
                  setQ(e.target.value);
                  setPage(1);
                }}
                placeholder="Search products, segments, strategies…"
                className="pl-9"
              />
            </div>
            <FilterSelect
              value={segment}
              onChange={(v) => {
                setSegment(v);
                setPage(1);
              }}
              options={segments}
              allLabel="All segments"
            />
            <FilterSelect
              value={strategy}
              onChange={(v) => {
                setStrategy(v);
                setPage(1);
              }}
              options={strategies}
              allLabel="All strategies"
            />
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4 text-muted-foreground" />
              <select
                value={sortKey}
                onChange={(e) => setSortKey(e.target.value as SortKey)}
                className="h-9 rounded-md border border-input bg-background px-3 text-sm"
              >
                <option value="score">Sort: Score</option>
                <option value="product">Sort: Product</option>
                <option value="segment">Sort: Segment</option>
                <option value="strategy">Sort: Strategy</option>
              </select>
            </div>
          </div>
          {!isLoading && (
            <div className="text-xs text-muted-foreground mt-3">
              {filtered.length} {filtered.length === 1 ? "recommendation" : "recommendations"}
            </div>
          )}
        </CardContent>
      </Card>

      {isLoading ? (
        <Loading rows={6} />
      ) : slice.length === 0 ? (
        <Card>
          <CardContent className="py-16 text-center text-muted-foreground">
            No campaigns match your filters.
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
            {slice.map((c, i) => (
              <CampaignCard key={i} c={c} />
            ))}
          </div>

          {totalPages > 1 && (
            <div className="flex items-center justify-between text-xs text-muted-foreground pt-6">
              <div>
                Showing {(safePage - 1) * pageSize + 1}–
                {Math.min(safePage * pageSize, filtered.length)} of {filtered.length}
              </div>
              <div className="flex gap-1">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={safePage <= 1}
                  className="px-3 py-1 rounded-md border border-border disabled:opacity-40 hover:bg-accent"
                >
                  Prev
                </button>
                <div className="px-3 py-1">
                  Page {safePage} / {totalPages}
                </div>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={safePage >= totalPages}
                  className="px-3 py-1 rounded-md border border-border disabled:opacity-40 hover:bg-accent"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}

function FilterSelect({
  value,
  onChange,
  options,
  allLabel,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  allLabel: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="h-9 rounded-md border border-input bg-background px-3 text-sm min-w-[160px]"
    >
      <option value="all">{allLabel}</option>
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}
