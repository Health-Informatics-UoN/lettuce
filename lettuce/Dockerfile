FROM ubuntu:22.04 AS builder
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes pipx

RUN pipx install uv 
ENV UV_CACHE_DIR=/tmp/uv_cache \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /src
COPY pyproject.toml ./
RUN uv sync 

FROM ubuntu:22.04
ENV PATH="/src/.venv/bin:$PATH"
WORKDIR src
COPY --from=builder /src/.venv /src/.venv
COPY . .
EXPOSE 8000

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
