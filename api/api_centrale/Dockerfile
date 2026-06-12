FROM python:3.11-slim

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Reproduction de la structure du repo pour que les chemins relatifs (../../libs) fonctionnent
WORKDIR /repo
COPY libs/ ./libs/
COPY api/api_centrale/requirements.txt ./api/api_centrale/

WORKDIR /repo/api/api_centrale

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Vérification que les modules sont bien installés
RUN python -c "import requests, fastapi, uvicorn, dotenv, fitz, s3fs, PyPDF2; print('All modules imported successfully')"

# Copie du code source
COPY api/api_centrale/ .

# Exposition du port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "main_centrale:app", "--host", "0.0.0.0", "--port", "8000"]