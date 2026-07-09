import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import {
  api,
  colorForSegment,
  INVENTORY_COLORS,
  INVENTORY_LEVELS,
  rowsForMetric,
  scoreOf,
  type InventoryLevel,
} from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { ApiError, Loading } from "@/components/ApiError";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export const Route = createFileRoute("/analytics")({
  component: AnalyticsPage,
});

function AnalyticsPage() {
  const campaigns = useQuery({ queryKey: ["campaigns"], queryFn: api.campaigns });
  const summary = useQuery({ queryKey: ["summary"], queryFn: api.summary });
  const candidateSummary = useQuery({
    queryKey: ["candidateSummary"],
    queryFn: api.candidateSummary,
  });

  const data = campaigns.data ?? [];

  const probBySegment = useMemo(() => {
    // Catalog-wide average (every scored candidate per segment), not just
    // the curated top-N `/campaigns` returns — purchase_probability is a
    // direct term in campaign_score, so the top-N is selected for
    // near-highest probability by construction and averages near 100% for
    // every segment regardless of real differences between segments.
    const rows = rowsForMetric(
      candidateSummary.data ?? [],
      "segment_purchase_probability"
    );
    return rows
      .map((r) => ({
        name: r.key,
        value: (r.avg_purchase_probability ?? 0) * 100,
      }))
      .sort((a, b) => b.value - a.value);
  }, [candidateSummary.data]);

  const inventoryDist = useMemo(() => {
    // Catalog-wide distribution (every scored candidate), not just the
    // curated top-N `/campaigns` returns — that sample skews toward
    // high-inventory items since inventory_score directly boosts ranking.
    const rows = rowsForMetric(candidateSummary.data ?? [], "inventory_distribution");
    const counts: Record<InventoryLevel, number> = { low: 0, medium: 0, high: 0 };
    rows.forEach((r) => {
      if (r.key in counts) counts[r.key as InventoryLevel] = r.count;
    });
    return INVENTORY_LEVELS.map((lvl) => ({
      level: lvl,
      label: lvl.charAt(0).toUpperCase() + lvl.slice(1),
      count: counts[lvl],
      fill: INVENTORY_COLORS[lvl],
    }));
  }, [candidateSummary.data]);

  const segmentScores = useMemo(() => {
    const buckets: Record<string, { sum: number; n: number }> = {};
    data.forEach((c) => {
      const seg = (c.customer_segment as string) ?? "Unknown";
      if (!buckets[seg]) buckets[seg] = { sum: 0, n: 0 };
      buckets[seg].sum += scoreOf(c);
      buckets[seg].n += 1;
    });
    return Object.entries(buckets)
      .map(([name, { sum, n }]) => ({ name, score: n ? sum / n : 0, count: n, fill: colorForSegment(name) }))
      .sort((a, b) => b.score - a.score);
  }, [data]);

  return (
    <>
      <PageHeader
        title="Analytics"
        description="Campaign and segment performance distilled from the FastAPI dataset."
      />

      {campaigns.error && <ApiError error={campaigns.error} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartCard title="Avg purchase probability by segment">
          {candidateSummary.isLoading ? (
            <Loading rows={3} />
          ) : candidateSummary.error ? (
            <ApiError error={candidateSummary.error} />
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={probBySegment} layout="vertical" margin={{ left: 30 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis type="number" stroke="var(--muted-foreground)" fontSize={12} tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
                <YAxis dataKey="name" type="category" stroke="var(--muted-foreground)" fontSize={12} width={130} />
                <Tooltip
                  contentStyle={chartTooltip}
                  formatter={(v: number) => [`${v.toFixed(1)}%`, "Avg prob."]}
                />
                <Bar dataKey="value" fill="var(--chart-1)" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </ChartCard>

        <ChartCard title="Inventory distribution">
          {candidateSummary.isLoading ? (
            <Loading rows={3} />
          ) : candidateSummary.error ? (
            <ApiError error={candidateSummary.error} />
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={inventoryDist}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="label" stroke="var(--muted-foreground)" fontSize={12} />
                <YAxis stroke="var(--muted-foreground)" fontSize={12} allowDecimals={false} />
                <Tooltip contentStyle={chartTooltip} formatter={(v: number) => [v, "Candidates"]} />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {inventoryDist.map((row, i) => (
                    <Cell key={i} fill={row.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </ChartCard>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Segment-level recommendation performance</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            {campaigns.isLoading ? (
              <Loading rows={3} />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={segmentScores}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="name" stroke="var(--muted-foreground)" fontSize={12} interval={0} angle={-15} textAnchor="end" height={60} />
                  <YAxis stroke="var(--muted-foreground)" fontSize={12} tickFormatter={(v) => `${v}%`} />
                  <Tooltip contentStyle={chartTooltip} formatter={(value: number, name: string) => name === "Avg score" ? [`${value.toFixed(1)}%`, name] : [value, name]} />
                  <Bar dataKey="score" name="Avg score" radius={[6, 6, 0, 0]}>
                    {segmentScores.map((row, i) => (
                      <Cell key={i} fill={row.fill} />
                    ))}
                  </Bar>
                  <Bar dataKey="count" name="# recs" fill="var(--chart-3)" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Campaign summary feed</CardTitle>
          </CardHeader>
          <CardContent>
            {summary.error && <ApiError error={summary.error} />}
            {summary.isLoading && <Loading rows={3} />}
            {summary.data && summary.data.length > 0 && <SummaryTable rows={summary.data} />}
            {summary.data && summary.data.length === 0 && (
              <p className="text-sm text-muted-foreground">No summary data returned.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}

const chartTooltip = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  color: "var(--foreground)",
} as const;

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="h-72">{children}</CardContent>
    </Card>
  );
}

function SummaryTable({ rows }: { rows: Record<string, unknown>[] }) {
  const cols = Object.keys(rows[0]);
  return (
    <div className="overflow-x-auto -mx-2">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-muted-foreground border-b border-border">
            {cols.map((c) => (
              <th key={c} className="px-3 py-2 font-medium capitalize">
                {c.replace(/_/g, " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-border/60 hover:bg-muted/40">
              {cols.map((c) => (
                <td key={c} className="px-3 py-2 tabular-nums">
                  {formatCell(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
  return String(v);
}
