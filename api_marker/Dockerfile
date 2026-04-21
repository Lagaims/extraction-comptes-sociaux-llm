FROM python:3.13-slim

# installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Installation des dépendances système (plus importantes pour marker-pdf)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-fra \
    libreoffice \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Code source
COPY src/ ./src/

# installer le projet
RUN uv sync --frozen

# Exposition du port
EXPOSE 8001

# Vérification
RUN uv run python -c "import marker; print('Marker OK')"

# Commande de démarrage
CMD ["uv", "run", "uvicorn", "main_marker:app", "--host", "0.0.0.0", "--port", "8001", "--app-dir", "/app/src"]
