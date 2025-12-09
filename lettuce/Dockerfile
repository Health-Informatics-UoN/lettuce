# Builder stage 
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

ENV UV_CACHE_DIR=/tmp/uv_cache \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /src
COPY pyproject.toml ./
RUN uv sync --no-dev --extra llama-cpu \
    && rm -rf /tmp/uv_cache ~/.cache/pip \
    && uv pip uninstall torch \
    && uv pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Runtime stage 
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \ 
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/src/.venv/bin:$PATH"
ENV INFERENCE_TYPE="llama-cpp-python"
WORKDIR /src
COPY --from=builder /src/.venv /src/.venv

LABEL org.opencontainers.image.title=Lettuce \
      org.opencontainers.image.description=lettuce-dev-base  \
      org.opencontainers.image.vendor=University\ of\ Nottingham \
      org.opencontainers.image.url=https://github.com/Health-Informatics-UoN/lettuce/pkgs/container/lettuce \
      org.opencontainers.image.documentation=https://health-informatics-uon.github.io/lettuce/ \
      org.opencontainers.image.source=https://github.com/Health-Informatics-UoN/lettuce \
      org.opencontainers.image.licenses=MIT

COPY . .
EXPOSE 8000

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
