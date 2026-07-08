# Deployment

The backend and frontend deploy to separate services — Cloudflare Workers doesn't run a Python/pandas/FAISS stack, so the two halves need different hosts.

## Backend → Render

1. Push the repo to GitHub, then in Render: **New → Blueprint**, point it at the repo. Render picks up [`render.yaml`](https://github.com/Letitia-Chang/omni-retail-ai/blob/main/render.yaml) automatically (build: `pip install -r requirements.txt && PYTHONPATH=. python scripts/build_product_index.py`; start: `uvicorn backend.main:app`).
2. Set the `ANTHROPIC_API_KEY` secret in the Render dashboard (left blank in `render.yaml` on purpose — never commit real keys).
3. Note the deployed URL (e.g. `https://omni-retail-ai-backend.onrender.com`) — the frontend needs it next.

Free tier spins down after inactivity, so the first request after a while has a ~30–60s cold start — expected for a portfolio demo, not a bug.

**Known limitation:** product images won't load on the deployed backend — the H&M image set (~30GB) isn't in the repo. The dashboard already handles this gracefully (falls back to a placeholder icon per product), so nothing breaks; it's just photo-less in production. Images work fully when running locally after `scripts/download_data.py`.

## Frontend → Cloudflare Workers

Requires Node.js **v22+** (`wrangler` itself won't run on older versions) and a Cloudflare account.

```bash
cd frontend
echo "VITE_API_BASE_URL=https://your-render-backend.onrender.com" > .env   # the URL from the Render step
npm install
npx wrangler login    # one-time browser OAuth to your Cloudflare account
npm run build
npm run deploy        # wraps `wrangler deploy`
```

`VITE_API_BASE_URL` is inlined into the bundle at build time, so it must point at the deployed backend *before* `npm run build` — rebuild and redeploy if the backend URL ever changes.

## Operational notes

- **Rate limiting:** `/generate-copy` is a public endpoint that calls the paid Claude API, so it's rate-limited to 10 requests/minute per IP, with a 200/day global cap as a backstop against abuse spread across many IPs.
- **CORS:** the backend allows all origins (`allow_origins=["*"]`) — fine for a public read-mostly demo API, but worth tightening to a specific origin list in a real production setting.
- **No CI/CD yet** — deploys are manual (`git push` + Render blueprint sync + `wrangler deploy`). See [Limitations & Roadmap](limitations-and-roadmap.md).
