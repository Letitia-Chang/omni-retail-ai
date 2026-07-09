import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import {
  api,
  getProductImageUrl,
  getSessionAdCopies,
  type SeedAdCopy,
  type SessionAdCopy,
} from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { ApiError, Loading } from "@/components/ApiError";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Download, Image as ImageIcon } from "lucide-react";

export const Route = createFileRoute("/ad-copies")({
  component: AdCopiesPage,
});

type Row = {
  source: "Sample" | "This session";
  article_id: number;
  product_name: string;
  customer_segment: string;
  promotion_strategy: string;
  generated_copy: string;
  grounded_on: string[];
  generated_at: string;
};

function seedToRow(c: SeedAdCopy): Row {
  return {
    source: "Sample",
    article_id: c.article_id,
    product_name: c.product_name,
    customer_segment: c.customer_segment,
    promotion_strategy: c.promotion_strategy,
    generated_copy: c.generated_copy,
    grounded_on: c.grounded_on ? c.grounded_on.split("; ") : [],
    generated_at: c.generated_at,
  };
}

function sessionToRow(c: SessionAdCopy): Row {
  return {
    source: "This session",
    article_id: c.article_id,
    product_name: c.product_name,
    customer_segment: c.customer_segment,
    promotion_strategy: c.promotion_strategy,
    generated_copy: c.generated_copy,
    grounded_on: c.grounded_on,
    generated_at: c.generated_at,
  };
}

function toCsv(rows: Row[]): string {
  const header = [
    "source",
    "article_id",
    "product_name",
    "customer_segment",
    "promotion_strategy",
    "generated_copy",
    "grounded_on",
    "generated_at",
  ];
  const escape = (v: string) => `"${v.replace(/"/g, '""')}"`;
  const lines = rows.map((r) =>
    [
      r.source,
      String(r.article_id),
      r.product_name,
      r.customer_segment,
      r.promotion_strategy,
      r.generated_copy,
      r.grounded_on.join("; "),
      r.generated_at,
    ]
      .map(escape)
      .join(",")
  );
  return [header.map(escape).join(","), ...lines].join("\n");
}

function downloadCsv(rows: Row[]) {
  const blob = new Blob([toCsv(rows)], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "omniretail-ad-copies.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function AdCopiesPage() {
  const seed = useQuery({ queryKey: ["adCopies"], queryFn: api.adCopies });
  const [sessionRows, setSessionRows] = useState<SessionAdCopy[]>([]);

  useEffect(() => {
    // Re-reads on every mount, which covers navigating here from the
    // Generator page; the focus listener also catches switching back from
    // another tab where a generation happened.
    setSessionRows(getSessionAdCopies());
    const onFocus = () => setSessionRows(getSessionAdCopies());
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, []);

  const rows: Row[] = [
    ...sessionRows.map(sessionToRow),
    ...(seed.data ?? []).map(seedToRow),
  ];

  return (
    <>
      <PageHeader
        title="Ad Copies"
        description="A library of AI-generated campaign copy — a permanent sample set plus anything you've generated yourself this session."
        action={
          rows.length > 0 && (
            <button
              onClick={() => downloadCsv(rows)}
              className="inline-flex items-center gap-2 h-9 px-4 rounded-md border border-input bg-background text-sm font-medium hover:bg-accent transition-colors"
            >
              <Download className="h-4 w-4" /> Export CSV
            </button>
          )
        }
      />

      {seed.error && <ApiError error={seed.error} />}

      {sessionRows.length > 0 && (
        <p className="mb-4 text-xs text-muted-foreground">
          {sessionRows.length} generated this session — these live only in your browser and
          disappear on refresh. Everything else is a permanent sample set generated once and
          shipped with the app.
        </p>
      )}

      {seed.isLoading ? (
        <Loading rows={4} />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {rows.map((r, i) => (
            <AdCopyCard key={i} row={r} />
          ))}
        </div>
      )}

      {!seed.isLoading && rows.length === 0 && (
        <p className="text-sm text-muted-foreground">No ad copy yet.</p>
      )}
    </>
  );
}

function AdCopyCard({ row }: { row: Row }) {
  const [imgFailed, setImgFailed] = useState(false);
  const imageUrl = getProductImageUrl(row.article_id);

  return (
    <Card className="overflow-hidden flex flex-col">
      <div className="relative aspect-[16/9] w-full overflow-hidden bg-muted flex items-center justify-center">
        {imageUrl && !imgFailed ? (
          <img
            src={imageUrl}
            alt={row.product_name}
            loading="lazy"
            onError={() => setImgFailed(true)}
            className="absolute inset-0 h-full w-full object-cover"
          />
        ) : (
          <ImageIcon className="h-8 w-8 text-foreground/20" />
        )}
        <Badge
          className="absolute top-2 right-2 border-transparent shadow-sm"
          variant={row.source === "This session" ? "default" : "secondary"}
        >
          {row.source}
        </Badge>
      </div>
      <CardContent className="p-4 flex-1 flex flex-col gap-2">
        <div className="font-medium text-sm leading-snug">{row.product_name}</div>
        <div className="flex flex-wrap gap-1.5 text-[11px] text-muted-foreground">
          <span>{row.customer_segment}</span>
          <span>·</span>
          <span>{row.promotion_strategy}</span>
        </div>
        <p className="text-sm leading-relaxed text-foreground/85 mt-1">{row.generated_copy}</p>
        {row.grounded_on.length > 0 && (
          <p className="mt-auto pt-2 text-[11px] text-muted-foreground">
            Grounded on: {row.grounded_on.slice(0, 3).join(", ")}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
