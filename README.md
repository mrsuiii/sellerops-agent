# SellerOps Agent

Tool-using e-commerce agents on one marketplace, built on LangGraph and the
Claude API, with a multi-turn seller simulator, an evaluation harness, and a demo
fashion store ("Kala Studio").

```text
synthetic orders/stock/reviews (seeded) → seller tools → LangGraph agent (model ⇄ tools)
    → multi-turn eval (user simulator, outcome verifier, pass@k / pass^k)
```

Direction: a **seller agent** (sales analysis, restock, review insight, caption
fixes with human approval) first, then a **buyer agent** (search, recommend,
product detail, cart). The earlier Qwen 0.5B fine-tuning track is archived.

## Live demo

**Demo:** `https://<your-app-name>.fly.dev/seller` (deploy: [docs/deploy.md](docs/deploy.md)).
The shop is at `/`, the API docs at `/docs`.

What to try on the seller page:

1. **Review an agent proposal.** Two are waiting. You see a word-level diff, then
   approve it (the shop shows the new caption at once and you can revert), reject it
   with a reason (the agent drafts a revision within the allowed fields), or cancel.
2. **Ask the agent for a caption from scratch** ("+ Caption baru"). The agent looks
   at the product data and a product photo. When the caption needs something
   neither of them shows (material, fit, care), it asks you first instead of
   making it up. Your answers become product facts it remembers next time, and
   "I don't know" leads to a caption that says the data is missing.
3. **Check the audit trail.** Every proposal records the photo the model saw (by
   SHA-256) and the seller facts it used, and the audit log lists every step.

Evaluation of this workflow (11 scenarios, code checks plus a judge from another model
family, [docs/eval-caption-2026-11-27.md](docs/eval-caption-2026-11-27.md)):

| Agent | code checks pass@1 | code + judge pass@1 | pass^k | asks when (and only when) needed |
|---|---|---|---|---|
| Claude Haiku 5.5, k=3 | 1.00 | 0.91 | pass^3 0.73 | 33/33 |
| Nemotron 3 Ultra (no vision), k=1 | 0.82 | 0.64 | n/a | 10/11 |

The judge is not yet calibrated against human labels.

How it is built: a LangGraph `StateGraph` written node by node, without
`create_agent`. It uses `interrupt()` for the seller's decision and for the
agent's questions, a SQLite checkpointer so threads survive restarts, and a
`RetryPolicy` on the one LLM node. The service layer is the source of truth,
with an atomic apply, stale detection and append-only versions and audit log.
Claude Haiku runs behind a per-visitor budget.

## Repository layout

```text
src/sellerops/
├── seller/           # seller backend: shared analytics, SQLite, caption approval service + audit log, /api/seller
├── simulator/        # seller simulator: synthetic 90-day orders, stock, reviews; episodes (eval only)
├── harness/          # LangGraph agent graph (model ⇄ tools, Claude API); never imports evaluation/simulator
├── evaluation/       # seller tools, terminal tools, tasks, user simulators, turn loop, verifier, metrics
└── store/            # Kala Studio FastAPI backend + BM25 recommender
web/          # Kala Studio React app (Vite, TypeScript, Tailwind): storefront + seller page, built to web/dist and served by the store API
data/         # store catalog snapshot, seller_sim/ (dataset + eval tasks + manifests)
scripts/      # seller simulator generation, catalog import
tests/
```

## Quickstart

Requires Python 3.12+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # store + agent + eval + dev tools
cp .env.example .env               # fill in only the keys you need
```

Dependency groups (`pyproject.toml`):

| Extra | For |
| --- | --- |
| *(none)* | core: seller simulator, settings, catalog (+ python-dotenv) |
| `store` | Kala Studio API (FastAPI, uvicorn) |
| `agent` | LangGraph runtime |
| `eval` | evaluation harness (multi-turn runner, user simulators, optional NVIDIA judge) |
| `dev` | `store` + `eval` + pytest, ruff, pre-commit |

The existing Conda environment `llm-post-training` still works; run
`pip install -e ".[dev]"` inside it once.

## Common commands

```bash
make test     # web build, pytest, vitest, Playwright e2e
make lint     # ruff + TypeScript typecheck + ESLint
make web-dev  # React app dev server on :5173 (API proxied to :8000)
make store    # Kala Studio on http://127.0.0.1:8000 (API docs at /docs, seller page at /seller)
make store-fake    # same, a free fake model drafts the caption revisions
make docker   # build and run the store container
make test-live     # integration tests with the real LLM (costs money; needs .env)
make try-approval  # play the seller in the caption approval graph (real LLM)
                   # free dry run: python scripts/try_caption_approval.py --fake
make studio        # LangGraph Studio for the workflows (make studio-fake: no LLM costs)
```

Other entry points:

```bash
make sim      # python scripts/generate_seller_sim.py: regenerate data/seller_sim/ (deterministic)
```

An eval CLI for the seller agent comes with Session A3 (after the A1 graph).

## Configuration

Secrets and settings come from environment variables; see
[`.env.example`](.env.example). Never commit `.env`.

| Variable | Used by |
| --- | --- |
| `LANGSMITH_API_KEY` | optional LangSmith tracing |
| `SELLEROPS_JUDGE_MODEL` | optional LLM judge for final messages: model id on the NVIDIA endpoint (uses `LLM_API_KEY`, `LLM_BASE_URL`) |
| `SELLEROPS_LLM_PROVIDER` | hosted model provider: `anthropic` (default) or `openai_compatible` (NVIDIA by default) |
| `ANTHROPIC_API_KEY` | Claude API key, read by the `anthropic` SDK |
| `SELLEROPS_LLM_MODEL`, `SELLEROPS_LLM_EFFORT` | Claude model id (default `claude-haiku-5-5`) and effort (default `medium`) |
| `SELLEROPS_LLM_FALLBACK_MODELS` | comma list of fallback models (same provider), tried when the primary call fails |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | OpenAI-compatible endpoint (default base URL: NVIDIA `integrate.api.nvidia.com/v1`) |
| `SELLEROPS_PROJECT_ROOT` | repo root (needed for non-editable installs, e.g. Docker) |
| `SELLEROPS_DATA_DIR`, `SELLEROPS_WEB_DIR`, `SELLEROPS_RESULTS_DIR`, `SELLEROPS_CATALOG_PATH` | override data locations |
| `SELLEROPS_SELLER_DB` | seller backend SQLite file (default `var/seller.sqlite`, seeded from `data/seller_sim` on first start) |
| `SELLEROPS_IMAGES_DIR` | prepared product images the model saw, by store and SHA-256 (default `var/images`) |
| `SELLEROPS_LLM_VISION` | `1` if the `openai_compatible` model reads images (Claude always does) |
| `SELLEROPS_APPROVALS_DB` | checkpoints of the caption approval workflow (default `var/approvals.sqlite`) |
| `SELLEROPS_LOG_LEVEL`, `SELLEROPS_LOG_FORMAT` | `INFO`/`DEBUG`/…, `text` or `json` |
| `SELLEROPS_CORS_ORIGINS` | comma-separated origins allowed to call the store API |
| `SELLEROPS_DEMO_MODE` | `1` for the public demo: LLM budget on, demo proposals seeded at startup |
| `SELLEROPS_DEMO_LLM_PER_CLIENT_HOUR`, `SELLEROPS_DEMO_LLM_PER_DAY` | demo LLM budget (defaults 10 per visitor per hour, 200 per day) |
| `SELLEROPS_CLIENT_IP_HEADER` | header the proxy sets to the client IP (`Fly-Client-IP` on Fly.io) |
| `SELLEROPS_APPROVAL_FAKE` | `1`: a free fake model drafts captions (`make store-fake`) |

Code reads these through `sellerops.settings.get_settings()`; a `.env` in the
project root is loaded automatically (real environment variables win).
Logging is configured once per entry point via `sellerops.logging_config.configure_logging()`;
the Docker image logs JSON lines and each store response carries an `x-request-id`.

## Seller backend

`make store` also serves the seller API under `/api/seller` (see `/docs`): sales,
product performance, stock and reviews for the seller UI, and the caption approval
flow. Everything goes through `sellerops.seller.SellerService` on SQLite:

```text
pending ──approve──► applied ──revert──► reverted      (approve = apply, in one transaction)
   │  ├──reject(feedback)──► rejected ──► revision (revision_of, max 5 rejections)
   │  └──cancel──► cancelled
   └──(another proposal applied first)──► stale
```

- Every caption the shop shows has a row in `caption_versions`; the schema refuses
  a caption change without one, and `audit_log` / `caption_versions` are append-only.
- Revert restores the previous caption as a new version, and only while the product
  still shows that proposal's caption.
- The seller agent's tools use `InProcessSellerClient`: read tools plus
  `propose_caption_update` only, so the agent can never apply a caption.
- Writes take an `X-Seller-User` header as the actor. **No authentication yet**: demo only.
- Reset the demo data: delete `var/seller.sqlite` (and `var/approvals.sqlite`) and restart.
- With the agent runtime installed, the approval **workflow** runs behind
  `GET /caption-proposals/{id}/approval` and `POST …/approval/decision`
  (`{"decision": "approve" | "reject" | "cancel", "feedback", "add_fields"}`):
  a reject drafts the revision with the LLM and the response already holds it;
  `POST …/approval/retry` re-runs a failed step. Graph checkpoints live in
  `var/approvals.sqlite` (`SELLEROPS_APPROVALS_DB`). The plain approve / reject /
  cancel routes stay as service calls without the graph.
- C3b: the draft sees a product photo (fetched by the backend from the catalog host,
  shrunk, stored by SHA-256; the model never gets a URL) when the model has vision.
  `POST /caption-requests` asks for a caption from scratch. The graph may stop with a
  question (`waiting_for: "answer"`), answered by `POST …/approval/answer`. Answers are
  stored as append-only product facts (`/products/{id}/facts`), and each proposal
  records `inputs.images` and `inputs.seller_facts`.
- One store today, but ready for more: each `SellerService` is bound to a `store_id`
  (the database records its owner and refuses another), callers get a service only via
  `SellerServices.for_store(store_id)` (`seller/registry.py`), and the API decides the
  store and the actor in two dependencies (`current_store_id`, `actor`). Going
  multi-store means a new `SellerServices` implementation and reading both from auth.

## Docker

One `Dockerfile`, two targets, so evaluation never shares an environment with production:

| Target | Image | Contains | Secrets |
| --- | --- | --- | --- |
| `store` (default) | `sellerops-store` | store + seller API + caption approval agent, catalog, seller demo data, web frontend (`[store,agent]`, pinned by `uv.lock`); `var/` on the `seller-db` volume | `ANTHROPIC_API_KEY` from the runtime env (Fly secret) |
| `eval` | `sellerops-eval` | agent + eval harness, simulator data and tasks (`[eval]` extra, pinned by `uv.lock`) | only from `.env.eval` at run time |

```bash
make docker                 # store on :8000 (docker compose up store also works)
make docker-demo            # the public demo locally: demo mode + free fake model
make deploy                 # lint, test, fly deploy --ha=false (docs/deploy.md)

cp .env.eval.example .env.eval   # fill in eval-only keys; never committed
make docker-eval-smoke      # dev tasks x1, scripted user
make docker-eval            # dev tasks x3, LLM user, NVIDIA judge
docker compose run --rm eval --task omzet --k 1   # any evaluate_seller.py flags
```

Neither image contains a `.env`. The eval service sits behind a compose profile, so
`docker compose up` never spends API credit; results land in `results/evaluation/`
through a volume, and `LANGSMITH_PROJECT=sellerops-eval` keeps its traces apart.
Both images run as a non-root user; the store exposes a health check at `/api/health`.

## Documentation

- [src/sellerops/evaluation/README.md](src/sellerops/evaluation/README.md): seller simulator, planted scenarios, multi-turn eval
- [web/README.md](web/README.md): Kala Studio store and API
- [docs/deploy.md](docs/deploy.md): public demo on Fly.io (budget, reset, secrets)
- [docs/eval-caption-2026-11-27.md](docs/eval-caption-2026-11-27.md): caption workflow eval (`scripts/evaluate_caption.py`)
- [PLAN.md](PLAN.md): learning plan and milestones
- The Qwen 0.5B track (code, weights, notebooks, results, old simulator) was removed on 17 Nov 2026;
  artifacts live in `../../archive/sellerops-sft-0.5b/` and the code in git history.

## Current status

- Seller simulator (C1) and multi-turn eval scaffolding are in place; 19 tasks (10 dev, 9 frozen test).
- Seller backend (C2 service + C3 caption approval backend) done: `src/sellerops/seller/`, `/api/seller`.
- Caption approval graph (C3) and C3b (product photos, agent asks the seller, product-fact memory) done: `src/sellerops/workflows/approval/`, seller page `/seller`.
- Agent graph (A1) done; seller tools in `harness/tools/seller.py`; eval CLI `scripts/evaluate_seller.py` (A3, first smoke run 8/10 dev tasks).
- No CI: run `make lint test` (or `pre-commit install` once) before pushing.
