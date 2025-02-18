FROM python:3.12-bookworm AS builder
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes pipx

RUN pipx install poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /src
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

FROM python:3.12-slim-bookworm
ENV PATH="/src/.venv/bin:$PATH"
WORKDIR src
COPY --from=builder /src/.venv /src/.venv
COPY . .
EXPOSE 8000

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
