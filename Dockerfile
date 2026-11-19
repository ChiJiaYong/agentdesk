# AgentDesk — multi-stage build (uv for fast, locked installs)
#
# Build:  docker build -t agentdesk .
# Run:    docker compose up   (first boot ingests the real datasets automatically)

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
# CPU-only torch keeps the image ~4x smaller than the CUDA default on Linux
RUN uv sync --frozen --extra dev --extra ml \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY . .
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com
EXPOSE 8000
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
