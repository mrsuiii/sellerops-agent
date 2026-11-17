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

## Repository layout

```text
src/sellerops/
├── simulator/        # seller simulator: synthetic 90-day orders, stock, reviews; read model; episodes (eval only)
├── harness/          # LangGraph agent graph (model ⇄ tools, Claude API); never imports evaluation/simulator
├── evaluation/       # seller tools, terminal tools, tasks, user simulators, turn loop, verifier, metrics
└── store/            # Kala Studio FastAPI backend + BM25 recommender
web/          # plain HTML/CSS/JS storefront served by the store API
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
| `eval` | evaluation harness (multi-turn runner, user simulators, optional Gemini judge) |
| `dev` | `store` + `eval` + pytest, ruff, pre-commit |

The existing Conda environment `llm-post-training` still works; run
`pip install -e ".[dev]"` inside it once.

## Common commands

```bash
make test     # pytest
make lint     # ruff
make store    # Kala Studio on http://127.0.0.1:8000 (API docs at /docs)
make docker   # build and run the store container
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
| `GEMINI_API_KEY` | optional Gemini prose judge |
| `SELLEROPS_LLM_PROVIDER` | hosted model provider: `anthropic` (default) or `openai_compatible` (adapter not wired yet) |
| `ANTHROPIC_API_KEY` | Claude API key, read by the `anthropic` SDK |
| `SELLEROPS_LLM_MODEL`, `SELLEROPS_LLM_EFFORT` | Claude model id (default `claude-haiku-5-5`) and effort (default `medium`) |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | OpenAI-compatible endpoint (default base URL: NVIDIA `integrate.api.nvidia.com/v1`) |
| `SELLEROPS_PROJECT_ROOT` | repo root (needed for non-editable installs, e.g. Docker) |
| `SELLEROPS_DATA_DIR`, `SELLEROPS_WEB_DIR`, `SELLEROPS_RESULTS_DIR`, `SELLEROPS_CATALOG_PATH` | override data locations |
| `SELLEROPS_LOG_LEVEL`, `SELLEROPS_LOG_FORMAT` | `INFO`/`DEBUG`/…, `text` or `json` |
| `SELLEROPS_CORS_ORIGINS` | comma-separated origins allowed to call the store API |

Code reads these through `sellerops.settings.get_settings()`; a `.env` in the
project root is loaded automatically (real environment variables win).
Logging is configured once per entry point via `sellerops.logging_config.configure_logging()`;
the Docker image logs JSON lines and each store response carries an `x-request-id`.

## Docker

```bash
docker build -t sellerops-store .
docker run --rm -p 8000:8000 sellerops-store
```

The image contains only the store API, its catalog snapshot and the web frontend.
It runs as a non-root user and exposes a health check at `/api/health`.

## Documentation

- [src/sellerops/evaluation/README.md](src/sellerops/evaluation/README.md): seller simulator, planted scenarios, multi-turn eval
- [web/README.md](web/README.md): Kala Studio store and API
- [PLAN.md](PLAN.md): learning plan and milestones
- The Qwen 0.5B track (code, weights, notebooks, results, old simulator) was removed on 17 Nov 2026;
  artifacts live in `../../archive/sellerops-sft-0.5b/` and the code in git history.

## Current status

- Seller simulator (C1) and multi-turn eval scaffolding are in place; 19 tasks (10 dev, 9 frozen test).
- The agent graph (`harness/state.py`, `harness/graph.py`) is Session A1 and still a TODO; its tests are skipped until then.
- No CI: run `make lint test` (or `pre-commit install` once) before pushing.
