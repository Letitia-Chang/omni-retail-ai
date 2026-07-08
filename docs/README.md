# Introduction

**OmniRetail AI** is an end-to-end retail marketing intelligence pipeline — from raw transaction data to a segmented, ranked, ready-to-ship campaign dashboard, with a live, retrieval-grounded LLM call generating the ad copy.

It's a portfolio project built to demonstrate applied data science and ML engineering across the full stack: customer segmentation, a supervised purchase-intent model, an inventory-aware campaign ranking system, a RAG pipeline grounding LLM-generated copy in a real product catalog, and a deployed FastAPI backend + React dashboard.

> Built for **AI/ML Engineer**, **Data Scientist**, and **Data Analyst** roles.

## Why this exists

Retail marketing teams manage thousands of products across multiple customer segments, but typically lack the tooling to plan campaigns at that scale:

- Campaign planning is manual and doesn't scale with catalog size or customer diversity.
- Promotions are rarely tied to actual purchase likelihood or current inventory, wasting spend on products customers won't buy or that are already out of stock.
- Customer segments exist as a concept ("loyal," "budget-conscious") but aren't operationalized into an actual targeting workflow.

OmniRetail AI turns raw customer/transaction/product data into a ranked list of *"which product, to which segment, with what strategy"* — the concrete output a marketing team would act on, plus the generated ad copy to go with it.

## What's in this book

- **[Key Features](features.md)** and **[Tech Stack](tech-stack.md)** — what the system does and what it's built with
- **[Architecture](architecture.md)** — how data flows from raw transactions to a deployed dashboard
- **[Getting Started](getting-started.md)** — clone, install, run the pipeline, run the app locally
- **[Demo Walkthrough](demo.md)** — a tour of the five dashboard views
- **[Deployment](deployment.md)** — how this is actually deployed (Render + Cloudflare Workers)
- **[Customer Segment Strategy](../reports/segment_strategy.md)** and **[Product Vision & Use Cases](../reports/use_cases.md)** — the domain thinking behind the segments and the target use cases
- **[Model & System Evaluation](../reports/evaluation.md)** — real, computed metrics for every model and pipeline stage, not estimates
- **[Limitations & Roadmap](limitations-and-roadmap.md)** — what's deliberately simplified, and what's next

## Live demo

- **Dashboard:** [tanstack-start-app.letitiachang0807-bb0.workers.dev](https://tanstack-start-app.letitiachang0807-bb0.workers.dev)
- **API:** [omni-retail-ai-backend.onrender.com](https://omni-retail-ai-backend.onrender.com)
- **Source:** [github.com/Letitia-Chang/omni-retail-ai](https://github.com/Letitia-Chang/omni-retail-ai)
