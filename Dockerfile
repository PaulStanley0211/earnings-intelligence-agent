# syntax=docker/dockerfile:1.7
# Multi-stage build. The builder pulls and resolves dependencies with uv,
# then the runtime stage copies just the virtual environment and the app
# source. Final image is small and runs as a non-root user.

ARG PYTHON_VERSION=3.11

# -----------------------------------------------------------------------------
# Builder stage: resolve and install dependencies via uv.
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Install uv. Pin to a known version for reproducibility.
COPY --from=ghcr.io/astral-sh/uv:0.4.27 /uv /uvx /usr/local/bin/

WORKDIR /build

# Copy only the files needed for dependency resolution first to maximize cache.
COPY pyproject.toml uv.lock README.md ./
COPY app/__init__.py app/__init__.py

# Frozen install: refuses to update uv.lock at build time.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy the rest of the source after deps are settled.
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY prompts/ ./prompts/

# -----------------------------------------------------------------------------
# Runtime stage: minimal image with the prebuilt venv.
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

# Non-root user for the runtime process.
RUN groupadd --gid 1001 app \
    && useradd --uid 1001 --gid app --shell /bin/bash --create-home app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/app /app/app
COPY --from=builder /build/migrations /app/migrations
COPY --from=builder /build/prompts /app/prompts

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
