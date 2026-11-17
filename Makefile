.PHONY: install lint fix test store docker sim

install:        ## Core + store + eval + dev tools (no torch)
	pip install -e ".[dev]"

lint:
	ruff check src scripts tests

fix:
	ruff check --fix src scripts tests

test:
	pytest -q

sim:            ## Regenerate data/seller_sim/ (dataset + eval tasks)
	python scripts/generate_seller_sim.py

store:          ## Run the Kala Studio demo store on :8000
	uvicorn sellerops.store.api:app --reload

docker:
	docker build -t sellerops-store . && docker run --rm -p 8000:8000 sellerops-store
