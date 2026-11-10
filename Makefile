.PHONY: install install-all lint fix test store docker

install:        ## Core + store + eval + dev tools (no torch)
	pip install -e ".[dev]"

install-all:    ## Everything, including local HF models (torch)
	pip install -e ".[dev,local-model]"

lint:
	ruff check src scripts tests

fix:
	ruff check --fix src scripts tests

test:
	pytest -q

store:          ## Run the Kala Studio demo store on :8000
	uvicorn sellerops.store.api:app --reload

docker:
	docker build -t sellerops-store . && docker run --rm -p 8000:8000 sellerops-store
