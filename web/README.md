# Kala Studio web app

The storefront (`/`, `/p/<id>`) and the seller approval page (`/seller`, `/seller/<id>`).
React 19 + TypeScript + Vite + Tailwind CSS v4, with React Router for pages and TanStack Query
for the API. The FastAPI backend (`src/sellerops/store/`) serves the built app from `web/dist`.

## Run it

```bash
make web-install          # once: npm dependencies into web/node_modules
make store-fake           # terminal 1: API on :8000 with the free fake model
make web-dev              # terminal 2: UI on http://localhost:5173 (hot reload, /api proxied to :8000)
```

Production-like, from one process: `make web-build && make store` → http://127.0.0.1:8000
(`/seller` for the seller page, `/docs` for the API).

## Layout

```
src/
  api/          typed API hooks: shop.ts (catalog), seller.ts (queues, approval, writes), types.ts
  lib/          api client (ApiError, X-Seller-User), cart (localStorage), diff (word diff), format
  components/   header, cart drawer, product card, toast, drawer, error boundary
  pages/shop/   catalog with URL-backed filters, product page with recommendations
  pages/seller/ queue, proposal and caption-request detail, question form, new caption drawer
```

Filters and the seller's status tab live in the URL, so a view can be shared and survives a reload.

## Checks

```bash
make lint      # ruff + TypeScript typecheck (tsc -b)
make test      # web build, Python tests (incl. the served app), vitest unit tests
npm run build  # production build only
```

## Notes for the demo

- There is no login. The seller name (top right on `/seller`) is sent as `X-Seller-User`
  on every write. The backend trusts it: fine for a demo, not for real store data.
- A demo budget limits agent steps per visitor (HTTP 429, `llm_budget_exhausted`); the
  page shows the remaining steps and turns the 429 into a toast.
- The caption language (English shop vs Indonesian drafts) is still open; see
  `docs/eval-caption-2026-11-27.md`.
