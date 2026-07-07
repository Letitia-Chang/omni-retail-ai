import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import {
  api,
  campaignMessageOf,
  copyAngleOf,
  explanationOf,
  probabilityOf,
  productOf,
  scoreOf,
  strategyOf,
  type Campaign,
} from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { ApiError, Loading } from "@/components/ApiError";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Sparkles, RefreshCw, Wand2 } from "lucide-react";

export const Route = createFileRoute("/generator")({
  component: Generator,
});

function Generator() {
  const segments = useQuery({ queryKey: ["segments"], queryFn: api.segments });
  const [segment, setSegment] = useState<string>("");

  useEffect(() => {
    if (!segment && segments.data?.segments?.[0]) setSegment(segments.data.segments[0]);
  }, [segments.data, segment]);

  const campaigns = useQuery({
    queryKey: ["campaigns", "segment", segment],
    queryFn: () => api.campaignsBySegment(segment),
    enabled: !!segment,
  });

  const products = useMemo(
    () => (campaigns.data ?? []).slice().sort((a, b) => scoreOf(b) - scoreOf(a)),
    [campaigns.data]
  );

  const [productIdx, setProductIdx] = useState(0);
  useEffect(() => setProductIdx(0), [segment]);

  const product = products[productIdx];
  const [seed, setSeed] = useState(0);

  return (
    <>
      <PageHeader
        title="AI Campaign Generator"
        description="Spin up on-brand campaign messages from the model's top recommendations."
      />

      {segments.error && <ApiError error={segments.error} />}

      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
        <Card className="h-fit">
          <CardContent className="p-5 space-y-4">
            <div>
              <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Customer segment
              </label>
              <select
                value={segment}
                onChange={(e) => setSegment(e.target.value)}
                className="mt-1.5 w-full h-9 rounded-md border border-input bg-background px-3 text-sm"
              >
                {segments.data?.segments.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Product
              </label>
              <select
                value={productIdx}
                onChange={(e) => setProductIdx(Number(e.target.value))}
                className="mt-1.5 w-full h-9 rounded-md border border-input bg-background px-3 text-sm"
                disabled={products.length === 0}
              >
                {products.map((p, i) => (
                  <option key={i} value={i}>
                    {productOf(p)} · {scoreOf(p).toFixed(1)}%
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={() => setSeed((s) => s + 1)}
              className="w-full inline-flex items-center justify-center gap-2 h-10 rounded-md bg-gradient-to-r from-primary to-chart-2 text-primary-foreground text-sm font-medium hover:opacity-90 transition-opacity shadow-sm"
            >
              <RefreshCw className="h-4 w-4" /> Regenerate
            </button>
          </CardContent>
        </Card>

        <div className="min-w-0">
          {campaigns.isLoading && <Loading rows={4} />}
          {!campaigns.isLoading && product && (
            <GeneratedOutput key={`${segment}-${productIdx}-${seed}`} product={product} segment={segment} />
          )}
        </div>
      </div>
    </>
  );
}

const HOOKS = [
  "Made for the way you shop",
  "Limited drop. Don't sleep on it.",
  "The upgrade your routine has been missing",
  "What everyone's reaching for this week",
  "Your next favorite, on the house*",
  "Engineered for the moments that matter",
];

const CTAS = ["Shop now", "See the drop", "Claim yours", "Add to bag", "Unlock the deal"];

function GeneratedOutput({ product, segment }: { product: Campaign; segment: string }) {
  const angle = copyAngleOf(product);
  const prob = probabilityOf(product);

  const hook = useMemo(() => HOOKS[Math.floor(Math.random() * HOOKS.length)], [product.article_id]);
  const cta = useMemo(() => CTAS[Math.floor(Math.random() * CTAS.length)], [product.article_id]);
  const headline = `${hook} — ${productOf(product)}`;

  // Live, RAG-grounded generation: retrieves similar catalog products via the
  // FAISS index and asks Claude to write copy from that context. Falls back
  // to the pre-computed template (campaign_message) if the call fails — e.g.
  // no ANTHROPIC_API_KEY configured — so the page still works without a key.
  const liveCopy = useMutation({
    mutationFn: () =>
      api.generateCopy(Number(product.article_id), segment, strategyOf(product)),
  });

  useEffect(() => {
    liveCopy.mutate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [product.article_id]);

  const fallbackBody =
    campaignMessageOf(product) ||
    explanationOf(product) ||
    `Built for ${segment}. ${strategyOf(product)} pricing live now.`;

  const body = liveCopy.data?.copy || fallbackBody;

  return (
    <div className="space-y-4">
      <Card className="border-primary/30 bg-gradient-to-br from-background to-primary/5">
        <CardHeader>
          <div className="flex items-center gap-2 text-xs font-medium text-primary uppercase tracking-wide">
            <Sparkles className="h-3.5 w-3.5" />
            {liveCopy.isPending
              ? "Generating with Claude, grounded on similar products…"
              : liveCopy.isSuccess
                ? "AI-generated campaign — live, RAG-grounded"
                : "AI-generated campaign (fallback template)"}
          </div>
          <CardTitle className="text-2xl mt-2 leading-snug">{headline}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {liveCopy.isPending ? (
            <Loading rows={2} />
          ) : (
            <p className="text-base leading-relaxed text-foreground/90">{body}</p>
          )}
          {liveCopy.isError && (
            <p className="text-xs text-muted-foreground">
              Live generation unavailable ({(liveCopy.error as Error).message}) — showing the
              pre-computed template instead.
            </p>
          )}
          {liveCopy.isSuccess && liveCopy.data.grounded_on.length > 0 && (
            <p className="text-xs text-muted-foreground">
              Grounded on similar products: {liveCopy.data.grounded_on.slice(0, 3).join(", ")}
            </p>
          )}
          <div className="flex flex-wrap gap-2 items-center">
            <button className="inline-flex items-center gap-1.5 h-10 px-5 rounded-md bg-primary text-primary-foreground text-sm font-medium">
              {cta}
            </button>
            <Badge variant="outline">Segment: {segment}</Badge>
            <Badge variant="outline">Strategy: {strategyOf(product)}</Badge>
            <Badge variant="outline">Purchase prob. {prob.toFixed(1)}%</Badge>
            <Badge className="bg-chart-2/15 text-chart-2 border-transparent">
              Score {scoreOf(product).toFixed(1)}%
            </Badge>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <Wand2 className="h-4 w-4 text-primary" /> Copy angle
            </CardTitle>
          </CardHeader>
          <CardContent>
            {angle ? (
              <p className="text-sm leading-relaxed">{angle}</p>
            ) : (
              <p className="text-sm text-muted-foreground">
                No copy angle provided for this recommendation.
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Recommended strategy</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div>
              <div className="text-xs text-muted-foreground">Promotion</div>
              <div className="font-medium mt-0.5">{strategyOf(product)}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Why this product</div>
              <div className="text-muted-foreground leading-relaxed mt-0.5">
                {explanationOf(product) || "High predicted purchase intent for this segment."}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
