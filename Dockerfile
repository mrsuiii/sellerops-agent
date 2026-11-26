# Two images from one Dockerfile, built with --target:
#
#   store  Kala Studio demo store API + static frontend (production). No LLM keys.
#          docker build --target store -t sellerops-store .
#          docker run --rm -p 8000:8000 sellerops-store
#
#   eval   Seller agent evaluation (Claude API + optional NVIDIA judge). Secrets come
#          only from --env-file .env.eval at run time; results go to a mounted volume.
#          docker build --target eval -t sellerops-eval .
#          docker run --rm --env-file .env.eval -v "$PWD/results/evaluation:/app/results/evaluation" \
#            sellerops-eval --user scripted --k 1
#
# Isolation: each image installs only its own extra ([store] vs [eval]), no .env is
# copied into either (.dockerignore), and settings never find a .env inside the image
# (SELLEROPS_PROJECT_ROOT=/app has none). The store image cannot call an LLM; the eval
# image does not ship the web frontend or the store server.

# ---------------------------------------------------------------- eval
FROM python:3.12-slim AS eval

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    SELLEROPS_PROJECT_ROOT=/app

WORKDIR /app

# Dependencies first (cached layer), pinned by uv.lock; --frozen fails if the lock is stale.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra eval
# Then the package itself, installed non-editable.
COPY src ./src
RUN uv sync --frozen --no-editable --extra eval && rm -rf src

# Runtime assets under SELLEROPS_PROJECT_ROOT: catalog + simulator data and tasks.
COPY data/store/fashion_catalog.json ./data/store/
COPY data/seller_sim ./data/seller_sim
COPY scripts/evaluate_seller.py ./scripts/

ARG GIT_COMMIT=unknown
ENV SELLEROPS_GIT_COMMIT=$GIT_COMMIT

RUN useradd --create-home --uid 10001 app && mkdir -p results/evaluation && chown -R app:app /app
USER app

ENTRYPOINT ["python", "scripts/evaluate_seller.py"]
CMD ["--user", "scripted", "--k", "1", "--tag", "smoke"]

# ---------------------------------------------------------------- store (default target)
FROM python:3.12-slim AS store

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SELLEROPS_PROJECT_ROOT=/app \
    SELLEROPS_LOG_FORMAT=json

WORKDIR /app

# Install the package (non-editable); runtime assets are located via
# SELLEROPS_PROJECT_ROOT, not via the source tree.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[store]" && rm -rf src

# Runtime assets under SELLEROPS_PROJECT_ROOT. data/seller_sim seeds the seller
# database (var/seller.sqlite) on first start; mount a volume on /app/var to keep
# captions, proposals and the audit log across container restarts.
COPY data/store ./data/store
COPY data/seller_sim ./data/seller_sim
COPY web ./web

RUN useradd --create-home --uid 10001 app && mkdir -p var && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status == 200 else 1)"

CMD ["uvicorn", "sellerops.store.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
