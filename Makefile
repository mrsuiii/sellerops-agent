.PHONY: install lint fix test test-live try-approval studio studio-fake store docker sim eval-smoke eval docker-eval-build docker-eval-smoke docker-eval

install:        ## Core + store + eval + dev tools (no torch)
	pip install -e ".[dev]"

lint:
	ruff check src scripts tests

fix:
	ruff check --fix src scripts tests

test:
	pytest -q

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

store:          ## Run the Kala Studio demo store on :8000
	uvicorn sellerops.store.api:app --reload

docker:         ## Production store image on :8000
	docker build --target store -t sellerops-store . && docker run --rm -p 8000:8000 sellerops-store

docker-eval-build:  ## Eval image (secrets only from .env.eval at run time)
	docker compose build --build-arg GIT_COMMIT=$$(git rev-parse --short HEAD) eval

docker-eval-smoke: docker-eval-build  ## Eval in Docker: dev x1, scripted user
	docker compose run --rm eval

docker-eval: docker-eval-build  ## Eval in Docker: dev x3, LLM user, judge
	docker compose run --rm eval --user llm --k 3 --judge
