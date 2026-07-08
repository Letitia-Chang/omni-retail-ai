# Demo Walkthrough

With the backend and frontend both running (see [Getting Started](getting-started.md)), or against the [live deployment](README.md#live-demo):

## 1. Overview (`/`)

Total campaigns, segment count, average campaign score, top recommended products, and a catalog-wide campaign-strategy mix — computed across every scored candidate, not just the curated top picks, so the chart shows the real spread rather than a biased sample.

## 2. Segments (`/segments`)

Explore each customer segment and its top recommended products, with real product images from the H&M catalog (when running with the full image set locally).

## 3. Campaigns (`/campaigns`)

Search and filter the full ranked campaign list — by segment, by strategy, or by product name.

## 4. Analytics (`/analytics`)

Score distributions and inventory-level breakdowns across the full scored catalog.

## 5. Generator (`/generator`)

Walks through a single campaign recommendation with its ranking explanation, and generates **live, RAG-grounded ad copy** via Claude — click "Regenerate" to see a fresh FAISS retrieval and a fresh Claude call each time. Falls back to a pre-computed template if `ANTHROPIC_API_KEY` isn't set, so the page never breaks even without a key.

See [Model & System Evaluation](../reports/evaluation.md) for a real example of this retrieval-and-generation step, with the actual retrieved neighbors and generated copy shown.
