FROM python:3.12-bookworm AS builder
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes pipx
ENV PATH="/root/.local/bin:${PATH}"
RUN pipx install poetry
RUN pipx inject poetry poetry-plugin-bundle
WORKDIR /src
COPY . .
RUN poetry bundle venv --python=/usr/bin/python3 /venv

FROM python:3.12-slim-bookworm

COPY --from=builder /venv /venv
# Expose port 8000
EXPOSE 8000

# Run the application
CMD [ "python", "app.py"]
