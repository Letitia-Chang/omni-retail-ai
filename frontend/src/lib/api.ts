// API client for OmniRetail AI FastAPI backend.
//
// Schema source: notebook 06 → ranked_campaign_recommendations.csv.
// All score-like fields (`campaign_score`, `purchase_probability`,
// `inventory_score`, `segment_match_score`) are floats in the 0–1 range
// at the API boundary. The dashboard normalizes them to 0–100 for display
// via the helpers below.

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

// ---------- Enums (exact values produced by the backend) ----------

export const CUSTOMER_SEGMENTS = [
  "High-Value One-Time Buyers",
  "Inactive Budget Shoppers",
  "Engaged Budget Shoppers",
  "Regular Shoppers",
  "Loyal High-Value Customers",
] as const;
export type CustomerSegment = (typeof CUSTOMER_SEGMENTS)[number];

export const PROMOTION_STRATEGIES = [
  "Promote aggressively",
  "Discount campaign",
  "Personalized recommendation",
  "Awareness campaign",
  "Premium positioning",
  "Deprioritize",
] as const;
export type PromotionStrategy = (typeof PROMOTION_STRATEGIES)[number];

export const INVENTORY_LEVELS = ["low", "medium", "high"] as const;
export type InventoryLevel = (typeof INVENTORY_LEVELS)[number];

// Numeric mapping used by the ranking notebook (low=0.3, medium=0.7, high=1.0).
export const INVENTORY_SCORE_MAP: Record<InventoryLevel, number> = {
  low: 0.3,
  medium: 0.7,
  high: 1.0,
};

// ---------- Deterministic color tokens for badges & charts ----------
// Resolve to existing CSS variables in src/styles.css so charts and badges
// share one palette.

export const SEGMENT_COLORS: Record<CustomerSegment, string> = {
  "High-Value One-Time Buyers": "var(--chart-1)",
  "Inactive Budget Shoppers": "var(--chart-2)",
  "Engaged Budget Shoppers": "var(--chart-3)",
  "Regular Shoppers": "var(--chart-4)",
  "Loyal High-Value Customers": "var(--chart-5)",
};

export const STRATEGY_COLORS: Record<PromotionStrategy, string> = {
  "Promote aggressively": "var(--chart-1)",
  "Discount campaign": "var(--chart-2)",
  "Personalized recommendation": "var(--chart-3)",
  "Awareness campaign": "var(--chart-4)",
  "Premium positioning": "var(--chart-5)",
  Deprioritize: "var(--muted-foreground)",
};

export const INVENTORY_COLORS: Record<InventoryLevel, string> = {
  low: "var(--chart-5)",
  medium: "var(--chart-3)",
  high: "var(--chart-2)",
};

export const colorForSegment = (s: string): string =>
  SEGMENT_COLORS[s as CustomerSegment] ?? "var(--muted-foreground)";

export const colorForStrategy = (s: string): string =>
  STRATEGY_COLORS[s as PromotionStrategy] ?? "var(--muted-foreground)";

// ---------- Row type ----------

export type Campaign = {
  customer_segment?: CustomerSegment | string;
  article_id?: number | string;
  product_name?: string;
  product_type?: string;
  product_group?: string;
  color_group?: string;
  style?: string;
  occasion?: string;
  target_audience?: string;
  purchase_probability?: number; // 0–1
  inventory_level?: InventoryLevel | string;
  inventory_score?: number; // 0–1
  segment_match_score?: number; // 0–1
  campaign_score?: number; // 0–1
  promotion_strategy?: PromotionStrategy | string;
  recommended_strategy?: string;
  product_copy_angle?: string;
  segment_copy_angle?: string;
  ranking_explanation?: string;
  campaign_message?: string;
  price_tier?: "Budget" | "Mid-range" | "Premium" | string;
};

// ---------- HTTP ----------

async function request<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} – ${path}`);
  return (await res.json()) as T;
}

async function requestPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `${res.status} ${res.statusText} – ${path}`);
  }
  return (await res.json()) as T;
}

export type GeneratedCopy = {
  article_id: number;
  copy: string;
  grounded_on: string[];
};

/** One row of the pre-generated Ad Copies sample library (see
 * scripts/generate_seed_ad_copy.py) — real Claude output, generated once
 * and committed as a static CSV, not created on demand. */
export type SeedAdCopy = {
  article_id: number;
  product_name: string;
  customer_segment: string;
  promotion_strategy: string;
  campaign_score: number; // 0–1
  purchase_probability: number; // 0–1
  generated_copy: string;
  grounded_on: string; // "; "-joined product names
  status: string;
  generated_at: string;
};

export const api = {
  segments: () => request<{ segments: string[] }>("/segments"),
  campaigns: () => request<Campaign[]>("/campaigns"),
  campaignsBySegment: (segment: string) =>
    request<Campaign[]>(`/campaigns/${encodeURIComponent(segment)}`),
  summary: () => request<Record<string, unknown>[]>("/summary"),
  candidateSummary: () =>
    request<CandidateSummaryRow[]>("/analytics/candidate-summary"),
  generateCopy: (articleId: number, customerSegment: string, promotionStrategy: string) =>
    requestPost<GeneratedCopy>("/generate-copy", {
      article_id: articleId,
      customer_segment: customerSegment,
      promotion_strategy: promotionStrategy,
    }),
  adCopies: () => request<SeedAdCopy[]>("/ad-copies"),
};

// ---------- Session-only ad copy history ----------
//
// Live "Regenerate" calls from the Generator page are never sent back to
// the backend to persist — that would mean a public write endpoint
// something could spam, and would make the sample library above grow
// without bound on every deploy. Instead each live generation is appended
// here (sessionStorage: cleared on tab close or refresh) and merged with
// the permanent sample library on the Ad Copies page.

export type SessionAdCopy = {
  article_id: number;
  product_name: string;
  customer_segment: string;
  promotion_strategy: string;
  generated_copy: string;
  grounded_on: string[];
  generated_at: string;
};

const SESSION_STORAGE_KEY = "omniretail-session-ad-copies";
const SESSION_MAX_ENTRIES = 50;

export const getSessionAdCopies = (): SessionAdCopy[] => {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as SessionAdCopy[]) : [];
  } catch {
    return [];
  }
};

/** Upsert by (article_id, customer_segment) rather than append — clicking
 * "Regenerate" on a product replaces its previous attempt in the session
 * list instead of piling up near-duplicate cards that differ only in copy
 * text, which is what a plain append produced. */
export const addSessionAdCopy = (entry: SessionAdCopy): SessionAdCopy[] => {
  const existing = getSessionAdCopies().filter(
    (c) => !(c.article_id === entry.article_id && c.customer_segment === entry.customer_segment)
  );
  const next = [entry, ...existing].slice(0, SESSION_MAX_ENTRIES);
  window.sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(next));
  return next;
};

// ---------- Catalog-wide candidate summary ----------
//
// `/analytics/candidate-summary` covers every scored (segment, product)
// candidate — not just the top-N recommendations `/campaigns` returns.
// The dashboard uses this for distribution charts (strategy mix, inventory
// distribution) so they reflect the whole catalog rather than a sample
// that's biased toward whatever the ranking formula favors.

export type CandidateSummaryMetric =
  | "strategy_mix"
  | "inventory_distribution"
  | "inventory_risk"
  | "segment_purchase_probability";

export type CandidateSummaryRow = {
  metric: CandidateSummaryMetric;
  key: string;
  count: number;
  avg_purchase_probability: number | null;
};

export const rowsForMetric = (
  rows: CandidateSummaryRow[],
  metric: CandidateSummaryMetric
): CandidateSummaryRow[] => rows.filter((r) => r.metric === metric);

// ---------- Field accessors ----------
//
// Always read campaign rows through these helpers — they keep field-name
// drift, type coercion, and 0–1 → 0–100 normalization in one place.

/**
 * Build an H&M product image URL from the article id.
 *
 * Source layout (see notebook data folder):
 *   data/raw/hm/images/{first 3 digits}/{10-digit article id}.jpg
 *
 * The FastAPI backend mounts that directory at `/images`, so the public URL is:
 *   {API_BASE_URL}/images/{folder}/{id}.jpg
 */
export const getProductImageUrl = (
  articleId: string | number | null | undefined
): string | null => {
  if (articleId == null) return null;
  const id = String(articleId).replace(/\D/g, "").padStart(10, "0");
  if (id.length !== 10) return null;
  const folder = id.slice(0, 3);
  return `${API_BASE_URL}/images/${folder}/${id}.jpg`;
};

/** Convenience wrapper that reads `article_id` straight from a campaign row. */
export const imageUrlOf = (c: Campaign): string | null =>
  getProductImageUrl(c.article_id);

export const productOf = (c: Campaign): string =>
  c.product_name ?? (c.article_id != null ? String(c.article_id) : "Unknown product");

export const strategyOf = (c: Campaign): string => c.promotion_strategy ?? "—";

export const explanationOf = (c: Campaign): string => c.ranking_explanation ?? "";

export const copyAngleOf = (c: Campaign): string => c.product_copy_angle ?? "";

export const campaignMessageOf = (c: Campaign): string => c.campaign_message ?? "";

export const inventoryLevelOf = (c: Campaign): InventoryLevel | null => {
  const raw = String(c.inventory_level ?? "").toLowerCase();
  return (INVENTORY_LEVELS as readonly string[]).includes(raw)
    ? (raw as InventoryLevel)
    : null;
};

// H&M's avg_selling_price is a pre-anonymized 0-1 index, not real currency,
// so this is a relative tier (ranked against the rest of the curated set)
// rather than a dollar figure — see campaign_generator.py for how it's computed.
const PRICE_TIER_SYMBOL: Record<string, string> = {
  Budget: "$",
  "Mid-range": "$$",
  Premium: "$$$",
};

export const priceTierOf = (c: Campaign): string | null =>
  c.price_tier ? (PRICE_TIER_SYMBOL[c.price_tier] ?? c.price_tier) : null;

/** campaign_score (0–1) → percentage (0–100). */
export const scoreOf = (c: Campaign): number => Number(c.campaign_score ?? 0) * 100;

/** purchase_probability (0–1) → percentage (0–100). */
export const probabilityOf = (c: Campaign): number =>
  Number(c.purchase_probability ?? 0) * 100;

// ---------- AI copilot suggestions ----------
//
// Deterministic suggestions derived from the recommended promotion strategy.
// These keep the "AI assistant" feel of the UI without inventing data —
// every output maps 1:1 from a known backend enum value.

export type CampaignChannel = "Email" | "Social" | "Push notification" | "In-app";

const STRATEGY_CTA: Record<PromotionStrategy, string> = {
  "Promote aggressively": "Shop the drop",
  "Discount campaign": "Claim your discount",
  "Personalized recommendation": "See your picks",
  "Awareness campaign": "Discover more",
  "Premium positioning": "Explore the collection",
  Deprioritize: "Browse alternatives",
};

const STRATEGY_CHANNEL: Record<PromotionStrategy, CampaignChannel> = {
  "Promote aggressively": "Push notification",
  "Discount campaign": "Email",
  "Personalized recommendation": "Email",
  "Awareness campaign": "Social",
  "Premium positioning": "Social",
  Deprioritize: "In-app",
};

const STRATEGY_ANGLE: Record<PromotionStrategy, string> = {
  "Promote aggressively": "Lead with urgency and scarcity — limited window, high intent buyers.",
  "Discount campaign": "Anchor on value — show original vs. discounted price and savings.",
  "Personalized recommendation": "Speak 1:1 — reference recent behavior and matching style.",
  "Awareness campaign": "Storytelling first — build the brand association before the offer.",
  "Premium positioning": "Emphasize craft, materials, and exclusivity over price.",
  Deprioritize: "Hold spend — surface only to warm audiences already browsing the category.",
};

export const ctaFor = (c: Campaign): string =>
  STRATEGY_CTA[c.promotion_strategy as PromotionStrategy] ?? "Shop now";

export const channelFor = (c: Campaign): CampaignChannel =>
  STRATEGY_CHANNEL[c.promotion_strategy as PromotionStrategy] ?? "Email";

export const marketingAngleFor = (c: Campaign): string =>
  copyAngleOf(c) ||
  STRATEGY_ANGLE[c.promotion_strategy as PromotionStrategy] ||
  "Highlight product fit for the segment.";

export const tagsOf = (c: Campaign): string[] => {
  const t = [c.style, c.occasion, c.product_group, c.color_group].filter(
    (v): v is string => typeof v === "string" && v.trim().length > 0
  );
  // de-dupe, preserve order
  return Array.from(new Set(t));
};
