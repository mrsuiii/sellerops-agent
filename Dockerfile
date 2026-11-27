# Two images from one Dockerfile, built with --target:
#
#   store  Kala Studio store (React app built in the web-build stage) + seller page + caption
#          approval agent (production, Fly.io).
#          docker build --target store -t sellerops-store .
#          docker run --rm -p 8000:8000 -e SELLEROPS_APPROVAL_FAKE=1 sellerops-store   # free fake model
#          docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY sellerops-store        # real agent
#
#   eval   Seller agent evaluation (Claude API + optional NVIDIA judge). Secrets come
#          only from --env-file .env.eval at run time; results go to a mounted volume.
#          docker build --target eval -t sellerops-eval .
#          docker run --rm --env-file .env.eval -v "$PWD/results/evaluation:/app/results/evaluation" \
#            sellerops-eval --user scripted --k 1
#
# Isolation: each image installs only its own extras ([store,agent] vs [eval]), no .env
# is copied into either (.dockerignore), and settings never find a .env inside the image
# (SELLEROPS_PROJECT_ROOT=/app has none). API keys come only from the runtime
# environment (fly secrets, --env-file). The eval image does not ship the web frontend
# or the store server.

# ---------------------------------------------------------------- web (build only; nothing from it ships but web/dist)
FROM node:22-alpine AS web-build

WORKDIR /web
# Lockfile first (cached layer); npm ci fails if package.json and the lock disagree.
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

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

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    SELLEROPS_PROJECT_ROOT=/app \
    SELLEROPS_LOG_FORMAT=json

WORKDIR /app

# [store] serves the shop; [agent] adds the approval graph (LangGraph, Claude, Pillow
# for product photos). Pinned by uv.lock; --frozen fails if the lock is stale.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra store --extra agent
COPY src ./src
RUN uv sync --frozen --no-editable --extra store --extra agent && rm -rf src

# Runtime assets under SELLEROPS_PROJECT_ROOT. data/seller_sim seeds the seller
# database (var/seller.sqlite) on first start. var/ also holds the approval
# checkpoints and the prepared product photos: mount a volume on /app/var to keep
# them across restarts (the public demo does not, so it starts fresh after idling).
COPY data/store ./data/store
COPY data/seller_sim ./data/seller_sim
COPY --from=web-build /web/dist ./web/dist

RUN useradd --create-home --uid 10001 app && mkdir -p var && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status == 200 else 1)"

CMD ["uvicorn", "sellerops.store.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
