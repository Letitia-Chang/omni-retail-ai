import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import {
  api,
  colorForSegment,
  colorForStrategy,
  productOf,
  rowsForMetric,
  scoreOf,
  strategyOf,
  type Campaign,
} from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { StatCard } from "@/components/StatCard";
import { ApiError, Loading } from "@/components/ApiError";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  AlertTriangle,
  ArrowRight,
  Sparkles,
  Target,
  TrendingUp,
  Users,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export const Route = createFileRoute("/overview")({
  component: Overview,
});

function Overview() {
  const campaigns = useQuery({ queryKey: ["campaigns"], queryFn: api.campaigns });
  const segments = useQuery({ queryKey: ["segments"], queryFn: api.segments });
  const candidateSummary = useQuery({
    queryKey: ["candidateSummary"],
    queryFn: api.candidateSummary,
  });

  const data: Campaign[] = campaigns.data ?? [];
  const avgScore =
    data.length > 0 ? data.reduce((sum, c) => sum + scoreOf(c), 0) / data.length : 0;

  const segmentCounts = countBy(data, (c) => (c.customer_segment as string) ?? "Unknown");

  // Top performing segments by AVG score (storytelling > raw counts)
  const segmentScores = useMemo(() => {
    const buckets: Record<string, { sum: number; n: number }> = {};
    data.forEach((c) => {
      const seg = (c.customer_segment as string) ?? "Unknown";
      if (!buckets[seg]) buckets[seg] = { sum: 0, n: 0 };
      buckets[seg].sum += scoreOf(c);
      buckets[seg].n += 1;
    });
    return Object.entries(buckets)
      .map(([name, { sum, n }]) => ({ name, score: n ? sum / n : 0, count: n }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 6);
  }, [data]);

  // "Active strategies" stat reflects the curated recommendations actually
  // shown (i.e. what's currently running).
  const strategyCounts = countBy(data, strategyOf);

  // Total catalog scored, not just the curated top-N — gives scale context
  // for what the model actually evaluated before narrowing down to the
  // picks shown below.
  const productsScored = useMemo(
    () =>
      rowsForMetric(candidateSummary.data ?? [], "strategy_mix").reduce(
        (sum, r) => sum + r.count,
        0
      ),
    [candidateSummary.data]
  );

  // The strategy-mix chart, on the other hand, shows the catalog-wide split
  // across every scored candidate, not just the curated top-N — that sample
  // skews toward whatever strategy the ranking formula favors (see
  // build_candidate_pool), so it isn't representative on its own.
  const strategyData = useMemo(
    () =>
      rowsForMetric(candidateSummary.data ?? [], "strategy_mix")
        .map((r) => ({ name: r.key, value: r.count }))
        .sort((a, b) => b.value - a.value),
    [candidateSummary.data]
  );

  // Top recommended products (highest scoring)
  const topProducts = useMemo(
    () =>
      data
        .slice()
        .sort((a, b) => scoreOf(b) - scoreOf(a))
        .slice(0, 5),
    [data]
  );

  // Inventory risk vs demand, computed catalog-wide (every scored candidate)
  // rather than from the curated top-N — the curated set is deliberately
  // biased toward high-inventory items, so it can't show stockout risk on
  // its own (there's nothing left in the "low stock" bucket to see).
  // High demand + low stock = stockout risk; low demand + high stock = overstock risk.
  const inventoryRisk = useMemo(() => {
    const rows = rowsForMetric(candidateSummary.data ?? [], "inventory_risk");
    const byLevel: Record<string, { demand: number; count: number }> = {};
    rows.forEach((r) => {
      byLevel[r.key] = {
        demand: (r.avg_purchase_probability ?? 0) * 100,
        count: r.count,
      };
    });
    return (["low", "medium", "high"] as const).map((lvl) => ({
      level: lvl,
      label: lvl === "low" ? "Low stock" : lvl === "medium" ? "Medium" : "High stock",
      demand: byLevel[lvl]?.demand ?? 0,
      count: byLevel[lvl]?.count ?? 0,
    }));
  }, [candidateSummary.data]);

  const stockoutRisk = inventoryRisk[0]; // low stock bucket
  const overstockRisk = inventoryRisk[2]; // high stock bucket

  return (
    <>
      <PageHeader
        title="Marketing command center"
        description="An at-a-glance view of where your AI assistant sees the highest opportunity right now."
      />

      {campaigns.error && <ApiError error={campaigns.error} />}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Products scored"
          value={productsScored.toLocaleString()}
          icon={Sparkles}
          loading={candidateSummary.isLoading}
          accent="primary"
          hint="Full catalog evaluated"
        />
        <StatCard
          label="Customer segments"
          value={segments.data?.segments.length ?? Object.keys(segmentCounts).length}
          icon={Users}
          loading={campaigns.isLoading}
          accent="chart-2"
        />
        <StatCard
          label="Avg campaign score"
          value={`${avgScore.toFixed(1)}%`}
          icon={TrendingUp}
          loading={campaigns.isLoading}
          hint={`Across ${data.length} curated picks`}
        />
        <StatCard
          label="Active strategies"
          value={Object.keys(strategyCounts).length}
          icon={Target}
          loading={campaigns.isLoading}
          accent="accent"
          hint="In today's curated picks"
        />
      </div>

      {/* Top picks + strategy mix */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mt-6">
        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Top recommended products</CardTitle>
              <p className="text-xs text-muted-foreground mt-1">
                Highest-scoring picks across every segment.
              </p>
            </div>
            <Link
              to="/segments"
              className="text-xs font-medium text-primary inline-flex items-center gap-1 hover:underline"
            >
              Explore <ArrowRight className="h-3 w-3" />
            </Link>
          </CardHeader>
          <CardContent>
            {campaigns.isLoading ? (
              <Loading rows={4} />
            ) : (
              <ul className="divide-y divide-border">
                {topProducts.map((p, i) => (
                  <li key={i} className="flex items-center gap-4 py-3 first:pt-0 last:pb-0">
                    <div className="flex h-8 w-8 items-center justify-center rounded-md bg-muted text-xs font-semibold tabular-nums text-muted-foreground">
                      {i + 1}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="font-medium truncate">{productOf(p)}</div>
                      <div className="text-xs text-muted-foreground truncate">
                        {(p.customer_segment as string) ?? "—"} · {strategyOf(p)}
                      </div>
                    </div>
                    <div className="hidden sm:block w-32">
                      <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full bg-primary"
                          style={{ width: `${Math.min(100, scoreOf(p))}%` }}
                        />
                      </div>
                    </div>
                    <div className="text-sm font-semibold tabular-nums w-12 text-right">
                      {scoreOf(p).toFixed(0)}%
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Campaign strategy mix</CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              How strategy breaks down across the full scored catalog.
            </p>
          </CardHeader>
          <CardContent className="h-72 flex flex-col">
            {candidateSummary.isLoading ? (
              <Loading rows={3} />
            ) : (
              <>
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={strategyData}
                        dataKey="value"
                        nameKey="name"
                        innerRadius={42}
                        outerRadius={75}
                        paddingAngle={2}
                      >
                        {strategyData.map((s, i) => (
                          <Cell key={i} fill={colorForStrategy(s.name)} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={chartTooltip} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <ul className="space-y-1 text-xs mt-2">
                  {strategyData.slice(0, 4).map((s) => (
                    <li key={s.name} className="flex items-center gap-2">
                      <span
                        className="h-2 w-2 rounded-full shrink-0"
                        style={{ background: colorForStrategy(s.name) }}
                      />
                      <span className="flex-1 truncate text-muted-foreground">{s.name}</span>
                      <span className="tabular-nums font-medium">{s.value}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Inventory risk + top segments */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mt-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-chart-5" />
              Inventory risk vs demand
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              Average purchase intent within each stock bucket, across the full catalog — high intent on low stock = stockout risk.
            </p>
          </CardHeader>
          <CardContent>
            {candidateSummary.isLoading ? (
              <Loading rows={3} />
            ) : (
              <>
                <div className="h-52">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={inventoryRisk}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="label" stroke="var(--muted-foreground)" fontSize={12} />
                      <YAxis
                        stroke="var(--muted-foreground)"
                        fontSize={12}
                        tickFormatter={(v) => `${v}%`}
                        domain={[0, 100]}
                      />
                      <Tooltip
                        contentStyle={chartTooltip}
                        formatter={(v: number, name: string) =>
                          name === "Demand" ? [`${v.toFixed(1)}%`, name] : [v, name]
                        }
                      />
                      <Bar dataKey="demand" name="Demand" radius={[6, 6, 0, 0]}>
                        {inventoryRisk.map((row, i) => (
                          <Cell
                            key={i}
                            fill={
                              row.level === "low"
                                ? "var(--chart-5)"
                                : row.level === "medium"
                                  ? "var(--chart-4)"
                                  : "var(--chart-2)"
                            }
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="grid grid-cols-2 gap-3 mt-4">
                  <RiskTile
                    tone="danger"
                    title="Stockout risk"
                    value={`${stockoutRisk?.count ?? 0} items`}
                    hint={`Avg intent ${stockoutRisk?.demand.toFixed(0) ?? 0}% on low stock`}
                  />
                  <RiskTile
                    tone="warn"
                    title="Overstock pressure"
                    value={`${overstockRisk?.count ?? 0} items`}
                    hint={`Avg intent ${overstockRisk?.demand.toFixed(0) ?? 0}% on high stock`}
                  />
                </div>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Top performing segments</CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              Ranked by average campaign score.
            </p>
          </CardHeader>
          <CardContent>
            {campaigns.isLoading ? (
              <Loading rows={4} />
            ) : (
              <ul className="space-y-3">
                {segmentScores.map((s) => (
                  <li key={s.name}>
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium truncate pr-2">{s.name}</span>
                      <span className="tabular-nums text-muted-foreground text-xs shrink-0">
                        <Badge variant="outline" className="font-normal mr-2">
                          {s.count}
                        </Badge>
                        <span className="font-semibold text-foreground">
                          {s.score.toFixed(0)}%
                        </span>
                      </span>
                    </div>
                    <div className="mt-1.5 h-1.5 rounded-full bg-muted overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.min(100, s.score)}%`,
                          background: colorForSegment(s.name),
                        }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
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

function RiskTile({
  tone,
  title,
  value,
  hint,
}: {
  tone: "danger" | "warn";
  title: string;
  value: string;
  hint: string;
}) {
  return (
    <div
      className={
        tone === "danger"
          ? "rounded-lg border border-chart-5/30 bg-chart-5/10 p-3"
          : "rounded-lg border border-chart-4/30 bg-chart-4/10 p-3"
      }
    >
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground font-medium">
        {title}
      </div>
      <div className="mt-1 font-semibold">{value}</div>
      <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div>
    </div>
  );
}


function countBy<T>(arr: T[], key: (t: T) => string): Record<string, number> {
  const out: Record<string, number> = {};
  arr.forEach((item) => {
    const k = key(item) || "Unknown";
    out[k] = (out[k] ?? 0) + 1;
  });
  return out;
}
