FROM python:3.12-bookworm AS builder
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes pipx

RUN pipx install uv 
ENV UV_CACHE_DIR=/tmp/uv_cache \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /src
COPY pyproject.toml ./
RUN uv sync --verbose \
    --extra-index-url https://pypi.org/simple \
    --build-option "llama-cpp-python:CMAKE_ARGS=-DGGML_NO_ACCELERATE=ON -DGGML_NO_OPENMP=ON -DGGML_NO_LLAMAFILE=ON"

FROM python:3.12-slim-bookworm
ENV PATH="/src/.venv/bin:$PATH"
WORKDIR src
COPY --from=builder /src/.venv /src/.venv
COPY . .
EXPOSE 8000

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
