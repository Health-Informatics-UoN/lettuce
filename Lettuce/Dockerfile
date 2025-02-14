FROM python:3.12-bookworm
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes pipx
ENV PATH="/root/.local/bin:${PATH}"
RUN pipx install poetry
RUN pipx inject poetry poetry-plugin-bundle
WORKDIR /src
COPY . .

RUN poetry install

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "python", "app.py"]
