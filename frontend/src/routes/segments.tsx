import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api, scoreOf, type Campaign } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { ApiError, Loading } from "@/components/ApiError";
import { CampaignCard } from "@/components/CampaignCard";
import { SegmentCard } from "@/components/SegmentCard";
import { Badge } from "@/components/ui/badge";
import { Sparkles } from "lucide-react";

export const Route = createFileRoute("/segments")({
  component: SegmentExplorer,
});

function SegmentExplorer() {
  const segments = useQuery({ queryKey: ["segments"], queryFn: api.segments });
  const allCampaigns = useQuery({ queryKey: ["campaigns"], queryFn: api.campaigns });
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    if (!selected && segments.data?.segments?.[0]) {
      setSelected(segments.data.segments[0]);
    }
  }, [segments.data, selected]);

  // Per-segment stats derived from full campaigns list
  const segmentStats = useMemo(() => {
    const map = new Map<string, { count: number; sum: number }>();
    (allCampaigns.data ?? []).forEach((c) => {
      const seg = (c.customer_segment as string) ?? "Unknown";
      const s = map.get(seg) ?? { count: 0, sum: 0 };
      s.count += 1;
      s.sum += scoreOf(c);
      map.set(seg, s);
    });
    return map;
  }, [allCampaigns.data]);

  const segmentCampaigns = useQuery({
    queryKey: ["campaigns", "segment", selected],
    queryFn: () => api.campaignsBySegment(selected!),
    enabled: !!selected,
  });

  const sortedRecs: Campaign[] = useMemo(
    () => (segmentCampaigns.data ?? []).slice().sort((a, b) => scoreOf(b) - scoreOf(a)),
    [segmentCampaigns.data]
  );

  return (
    <>
      <PageHeader
        title="Segment-first campaign explorer"
        description="Pick a customer segment to load AI-ranked product recommendations, generated copy, and channel suggestions in real time."
      />

      {segments.error && <ApiError error={segments.error} />}

      {/* Segment cards strip */}
      <section aria-label="Customer segments">
        {segments.isLoading ? (
          <Loading rows={2} />
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {segments.data?.segments.map((s) => {
              const stat = segmentStats.get(s);
              const count = stat?.count ?? 0;
              const avg = stat && stat.count ? stat.sum / stat.count : 0;
              return (
                <SegmentCard
                  key={s}
                  name={s}
                  count={count}
                  avgScore={avg}
                  active={selected === s}
                  onClick={() => setSelected(s)}
                />
              );
            })}
          </div>
        )}
      </section>

      {/* Recommendations */}
      <section className="mt-10" aria-label="Recommendations">
        {segmentCampaigns.error && <ApiError error={segmentCampaigns.error} />}

        {selected && (
          <div className="flex items-end justify-between gap-3 mb-5">
            <div>
              <div className="flex items-center gap-2 text-xs font-medium text-primary uppercase tracking-wide">
                <Sparkles className="h-3.5 w-3.5" />
                AI-ranked picks
              </div>
              <h2 className="mt-1 text-2xl font-semibold tracking-tight">{selected}</h2>
            </div>
            {segmentCampaigns.data && (
              <Badge variant="secondary" className="h-7 px-3">
                {segmentCampaigns.data.length} recommendations
              </Badge>
            )}
          </div>
        )}

        {segmentCampaigns.isLoading && <Loading rows={4} />}

        {sortedRecs.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
            {sortedRecs.map((c, i) => (
              <CampaignCard key={i} c={c} />
            ))}
          </div>
        )}

        {segmentCampaigns.data && sortedRecs.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No recommendations returned for this segment.
          </p>
        )}
      </section>
    </>
  );
}
