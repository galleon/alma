FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first so code-only changes don't bust the layer cache.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/docker-entrypoint.sh"]
