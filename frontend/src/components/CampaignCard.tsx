import { useState } from "react";
import {
  channelFor,
  colorForStrategy,
  copyAngleOf,
  ctaFor,
  explanationOf,
  imageUrlOf,
  inventoryLevelOf,
  marketingAngleFor,
  probabilityOf,
  productOf,
  scoreOf,
  strategyOf,
  tagsOf,
  type Campaign,
  type InventoryLevel,
} from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ChevronDown,
  Image as ImageIcon,
  Mail,
  Megaphone,
  Smartphone,
  Sparkles,
  Target,
  TrendingUp,
  Users,
  Wand2,
} from "lucide-react";
import { cn } from "@/lib/utils";

const INVENTORY_TONE: Record<InventoryLevel, string> = {
  low: "bg-chart-5/15 text-chart-5 border-chart-5/30",
  medium: "bg-chart-4/15 text-chart-4 border-chart-4/30",
  high: "bg-chart-2/15 text-chart-2 border-chart-2/30",
};

const CHANNEL_ICON = {
  Email: Mail,
  Social: Megaphone,
  "Push notification": Smartphone,
  "In-app": Sparkles,
} as const;

export function CampaignCard({ c }: { c: Campaign }) {
  const [open, setOpen] = useState(false);
  const score = scoreOf(c);
  const prob = probabilityOf(c);
  const inv = inventoryLevelOf(c);
  const strategy = strategyOf(c);
  const channel = channelFor(c);
  const cta = ctaFor(c);
  const angle = marketingAngleFor(c);
  const tags = tagsOf(c);
  const ChannelIcon = CHANNEL_ICON[channel];
  const imageUrl = imageUrlOf(c);
  const [imgFailed, setImgFailed] = useState(false);

  return (
    <Card className="group flex flex-col overflow-hidden transition-all hover:shadow-lg hover:-translate-y-0.5 hover:border-primary/40">
      {/* Product image with strategy color wash + hover zoom */}
      <div
        className="relative aspect-[4/5] w-full overflow-hidden rounded-t-lg flex items-center justify-center"
        style={{
          background: `linear-gradient(135deg, color-mix(in oklab, ${colorForStrategy(strategy)} 22%, var(--card)), var(--card))`,
        }}
      >
        {imageUrl && !imgFailed ? (
          <img
            src={imageUrl}
            alt={productOf(c)}
            loading="lazy"
            onError={() => setImgFailed(true)}
            className="absolute inset-0 h-full w-full object-cover transition-transform duration-500 ease-out group-hover:scale-105"
          />
        ) : (
          <ImageIcon className="h-10 w-10 text-foreground/20" />
        )}
        <div className="absolute top-3 right-3 rounded-md bg-background/80 backdrop-blur px-2 py-1 text-xs font-semibold tabular-nums shadow-sm">
          {score.toFixed(0)}% match
        </div>
        <div className="absolute bottom-3 left-3 flex flex-wrap gap-1.5">
          <Badge
            className="border-transparent text-foreground shadow-sm"
            style={{
              background: `color-mix(in oklab, ${colorForStrategy(strategy)} 70%, var(--background))`,
            }}
          >
            {strategy}
          </Badge>
          {inv && (
            <Badge variant="outline" className={cn("border bg-background/80 backdrop-blur", INVENTORY_TONE[inv])}>
              {inv} stock
            </Badge>
          )}
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-4 p-5">
        <div>
          <h3 className="font-semibold text-base leading-snug line-clamp-2">
            {productOf(c)}
          </h3>
          {c.target_audience && (
            <div className="mt-1 flex items-center gap-1.5 text-xs text-muted-foreground">
              <Users className="h-3 w-3" />
              <span className="truncate">{c.target_audience}</span>
            </div>
          )}
        </div>

        {/* Score / probability rail */}
        <div className="grid grid-cols-2 gap-2">
          <Stat
            icon={TrendingUp}
            label="Campaign score"
            value={`${score.toFixed(0)}%`}
            barValue={score}
            tone="primary"
          />
          <Stat
            icon={Target}
            label="Purchase prob."
            value={`${prob.toFixed(0)}%`}
            barValue={prob}
            tone="chart-2"
          />
        </div>

        {/* Generated copy line */}
        {copyAngleOf(c) && (
          <p className="text-sm leading-relaxed text-foreground/85 italic">
            “{copyAngleOf(c)}”
          </p>
        )}

        {/* Tags */}
        {tags.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {tags.slice(0, 5).map((t) => (
              <span
                key={t}
                className="text-[11px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground"
              >
                {t}
              </span>
            ))}
          </div>
        )}

        {/* Copilot panel */}
        <div className="mt-auto rounded-lg border border-border bg-muted/40">
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            className="w-full flex items-center justify-between gap-2 px-3 py-2.5 text-left"
            aria-expanded={open}
          >
            <span className="flex items-center gap-2 text-xs font-medium text-foreground">
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              AI copilot · why this works
            </span>
            <ChevronDown
              className={cn(
                "h-4 w-4 text-muted-foreground transition-transform",
                open && "rotate-180"
              )}
            />
          </button>
          {open && (
            <div className="px-3 pb-3 pt-1 space-y-3 text-sm">
              <Block icon={Sparkles} label="Why recommended">
                {explanationOf(c) ||
                  "Strong predicted purchase intent for this segment based on past behavior."}
              </Block>
              <Block icon={Wand2} label="Suggested marketing angle">
                {angle}
              </Block>
              <div className="grid grid-cols-2 gap-2">
                <MiniBlock label="Suggested CTA" value={cta} />
                <MiniBlock
                  label="Suggested channel"
                  value={
                    <span className="inline-flex items-center gap-1.5">
                      <ChannelIcon className="h-3.5 w-3.5" />
                      {channel}
                    </span>
                  }
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

function Stat({
  icon: Icon,
  label,
  value,
  barValue,
  tone,
}: {
  icon: typeof TrendingUp;
  label: string;
  value: string;
  barValue: number;
  tone: "primary" | "chart-2";
}) {
  return (
    <div className="rounded-md border border-border bg-background p-2.5">
      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1">
          <Icon className="h-3 w-3" /> {label}
        </span>
        <span className="font-semibold tabular-nums text-foreground">{value}</span>
      </div>
      <div className="mt-1.5 h-1 rounded-full bg-muted overflow-hidden">
        <div
          className={cn(
            "h-full rounded-full transition-all",
            tone === "primary" ? "bg-primary" : "bg-chart-2"
          )}
          style={{ width: `${Math.max(2, Math.min(100, barValue))}%` }}
        />
      </div>
    </div>
  );
}

function Block({
  icon: Icon,
  label,
  children,
}: {
  icon: typeof Sparkles;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        <Icon className="h-3 w-3" />
        {label}
      </div>
      <div className="mt-1 text-sm leading-relaxed text-foreground/85">{children}</div>
    </div>
  );
}

function MiniBlock({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-md border border-border bg-background px-2.5 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground font-medium">
        {label}
      </div>
      <div className="mt-0.5 text-sm font-medium text-foreground">{value}</div>
    </div>
  );
}
