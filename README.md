# SellerOps Agent

Tool-using e-commerce agents on one marketplace, with a deterministic simulator,
an evaluation harness, and a demo fashion store ("Kala Studio").

```text
deterministic simulator → frozen evaluation tasks → base-model evaluation
    → full SFT → LangGraph runtime + trajectory replay → (optional) DPO
```

Planned direction: a **buyer agent** (shopping assistant: search, recommend,
product detail, cart) and a **seller agent** (sales analysis and advice), backed
by an API-hosted model. The fine-tuned Qwen2.5-0.5B remains as a baseline.

## Repository layout

```text
src/sellerops/
├── schemas.py, environment.py, dataset.py   # action schemas, simulator, task generator (stdlib only)
├── services/episode_service/                # isolated episodes, snapshot + replay
├── agent/                                   # LangGraph router → action → tools runtime
├── evaluation/                              # verifier, LangSmith runner, metrics, judge
└── store/                                   # Kala Studio FastAPI backend + BM25 recommender
web/          # plain HTML/CSS/JS storefront served by the store API
data/         # tasks, SFT data, answer keys, store catalog snapshot
scripts/      # dataset generation, catalog import, evaluation CLI
notebooks/    # baseline and SFT notebooks (Colab)
results/      # evaluation and training summaries (weights are gitignored)
docs/         # deep-dive guides
tests/
```

## Quickstart

Requires Python 3.12+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # store + agent + eval + dev tools, no torch
# pip install -e ".[dev,local-model]"  # add torch/transformers for local HF models
cp .env.example .env               # fill in only the keys you need
```

Dependency groups (`pyproject.toml`):

| Extra | For |
| --- | --- |
| *(none)* | core: schemas, simulator, dataset, episode service, verifier |
| `store` | Kala Studio API (FastAPI, uvicorn) |
| `agent` | LangGraph runtime |
| `eval` | evaluation harness (LangSmith, Gemini judge) |
| `local-model` | local Qwen base / SFT checkpoint (torch, transformers) |
| `dev` | `store` + `eval` + pytest, ruff, pre-commit |

The existing Conda environment `llm-post-training` still works; run
`pip install -e ".[dev]"` inside it once.

## Common commands

```bash
make test     # pytest (torch-dependent tests are skipped without the local-model extra)
make lint     # ruff
make store    # Kala Studio on http://127.0.0.1:8000 (API docs at /docs)
make docker   # build and run the store container
```

Other entry points:

```bash
python scripts/generate_sellerops_dataset.py        # regenerate deterministic dataset
python scripts/evaluate_agent.py --offline --limit 1 # local eval (needs local-model + SFT weights)
python scripts/evaluate_agent.py --cloud            # publish to LangSmith (LANGSMITH_API_KEY)
```

`--judge gemini` (needs `GEMINI_API_KEY`) sends evaluation content to Google even
with `--offline`; `--offline` only controls LangSmith publishing.

## Configuration

Secrets and settings come from environment variables; see
[`.env.example`](.env.example). Never commit `.env`.

| Variable | Used by |
| --- | --- |
| `LANGSMITH_API_KEY` | `evaluate_agent.py --cloud` |
| `GEMINI_API_KEY` | optional Gemini prose judge |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | planned hosted-model adapter |
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

- [AGENT_WALKTHROUGH.md](AGENT_WALKTHROUGH.md): router/action graph walkthrough
- [src/sellerops/evaluation/README.md](src/sellerops/evaluation/README.md): evaluation harness
- [docs/episode-replay-and-verification.md](docs/episode-replay-and-verification.md): message conversion, snapshots, replay, outcome verifier
- [docs/sft-training.md](docs/sft-training.md): full SFT on Colab
- [web/README.md](web/README.md): Kala Studio store and API
- [data/DATASET_CARD.md](data/DATASET_CARD.md): dataset
- [PLAN.md](PLAN.md): learning plan and milestones

## Current status

- 116 torch-free tests pass; 3 test modules need the `local-model` extra.
- No CI: run `make lint test` (or `pre-commit install` once) before pushing.
- Full SFT of Qwen2.5-0.5B-Instruct: schema validity 17.9% → 100%, exact action 0% → 70.5% (dev).
- End-to-end dev eval is 0/30: the base-model router emits text on `act`, so the SFT action model is never called.
