FROM python:3.13-slim

# installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copie du code source
COPY src/ ./src/

# installer le projet
RUN uv sync --frozen

# Exposition du port
EXPOSE 1324

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:1324/health || exit 1

# Commande de démarrage
CMD ["uv", "run", "uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "1324", "--app-dir", "/app/src"]
