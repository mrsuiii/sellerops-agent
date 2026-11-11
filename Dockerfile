# Kala Studio demo store API + static frontend.
# Build: docker build -t sellerops-store .
# Run:   docker run --rm -p 8000:8000 sellerops-store
FROM python:3.12-slim AS runtime

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

# Runtime assets under SELLEROPS_PROJECT_ROOT.
COPY data/store ./data/store
COPY web ./web

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status == 200 else 1)"

CMD ["uvicorn", "sellerops.store.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
