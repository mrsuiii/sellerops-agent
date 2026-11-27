.PHONY: install lint fix test test-live try-approval studio studio-fake store store-fake web-install web-dev web-build web-lint web-test web-e2e docker docker-demo deploy sim eval-smoke eval docker-eval-build docker-eval-smoke docker-eval

install:        ## Core + store + eval + dev tools (no torch)
	pip install -e ".[dev]"

lint: web-install  ## Ruff, then the web app: TypeScript typecheck and ESLint
	ruff check src scripts tests
	cd web && npm run typecheck && npm run lint

fix:
	ruff check --fix src scripts tests

test: web-build  ## Web build, Python tests (incl. the served app), web unit tests, end-to-end
	pytest -q
	cd web && npm test && npm run test:e2e

test-live:      ## Integration tests against the real LLM (costs money; needs .env)
	SELLEROPS_LIVE_TESTS=1 pytest -q -m live -s tests/integration

try-approval:   ## Play the seller in the caption approval graph (real LLM; --fake for free)
	python scripts/try_caption_approval.py

sim:            ## Regenerate data/seller_sim/ (dataset + eval tasks)
	python scripts/generate_seller_sim.py

eval-smoke:     ## Claude API: dev tasks x1, scripted user (cheap)
	python scripts/evaluate_seller.py --user scripted --k 1 --tag smoke

eval:           ## Claude API (+ NVIDIA judge): dev tasks x3, LLM user
	python scripts/evaluate_seller.py --user llm --k 3 --judge

studio:         ## LangGraph Studio for the workflows (real LLM; needs: uv sync --extra dev --extra studio)
	langgraph dev --allow-blocking

studio-fake:    ## Same, with a free fake model writing the revisions
	SELLEROPS_STUDIO_FAKE=1 langgraph dev --allow-blocking

store:          ## Kala Studio on :8000 (API + the built web app if web/dist exists; revisions use the real LLM)
	uvicorn sellerops.store.api:app --reload

store-fake:     ## Same, with a free fake model drafting the caption revisions
	SELLEROPS_APPROVAL_FAKE=1 uvicorn sellerops.store.api:app --reload

web-install:    ## Install the web app dependencies (web/node_modules)
	cd web && npm install --no-audit --no-fund

web-dev:        ## Web app with hot reload on :5173 (API proxied to :8000; run make store-fake too)
	cd web && npm run dev

web-build: web-install  ## Production build of the web app into web/dist (served by make store)
	cd web && npm run build

web-test: web-install  ## Web unit tests (vitest)
	cd web && npm test

web-e2e: web-build  ## End-to-end tests in Chrome against the real store (fresh database)
	cd web && npm run test:e2e

web-lint: web-install  ## Typecheck and ESLint for the web app
	cd web && npm run typecheck && npm run lint

docker:         ## Production store image on :8000
	docker build --target store -t sellerops-store . && docker run --rm -p 8000:8000 sellerops-store

docker-demo:    ## The public demo locally (demo mode, free fake model) on :8000
	docker build --target store -t sellerops-store . && \
	docker run --rm -p 8000:8000 -e SELLEROPS_DEMO_MODE=1 -e SELLEROPS_APPROVAL_FAKE=1 sellerops-store

deploy:         ## Lint, test, then deploy the demo to Fly.io (one machine; see docs/deploy.md)
	$(MAKE) lint test && fly deploy --ha=false

docker-eval-build:  ## Eval image (secrets only from .env.eval at run time)
	docker compose build --build-arg GIT_COMMIT=$$(git rev-parse --short HEAD) eval

docker-eval-smoke: docker-eval-build  ## Eval in Docker: dev x1, scripted user
	docker compose run --rm eval

docker-eval: docker-eval-build  ## Eval in Docker: dev x3, LLM user, judge
	docker compose run --rm eval --user llm --k 3 --judge
