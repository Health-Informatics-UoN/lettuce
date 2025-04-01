# Builder stage 
FROM python:3.12-bookworm AS builder
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes pipx

RUN pipx install uv 
ENV UV_CACHE_DIR=/tmp/uv_cache \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /src
COPY pyproject.toml ./
RUN uv sync 

# Runtime stage 
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgcc-s1 \
    libgomp1 \ 
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/src/.venv/bin:$PATH"
WORKDIR /src
COPY --from=builder /src/.venv /src/.venv

LABEL org.opencontainers.image.title=Lettuce
LABEL org.opencontainers.image.description=lettuce-dev-base 
LABEL org.opencontainers.image.vendor=University\ of\ Nottingham
LABEL org.opencontainers.image.url=https://github.com/Health-Informatics-UoN/lettuce/pkgs/container/lettuce
LABEL org.opencontainers.image.documentation=https://health-informatics-uon.github.io/lettuce/
LABEL org.opencontainers.image.source=https://github.com/Health-Informatics-UoN/lettuce
LABEL org.opencontainers.image.licenses=MIT

COPY . .
EXPOSE 8000

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
