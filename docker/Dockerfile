# Dockerfile for API
# docker/Dockerfile
# Dockerfile pour l'API FastAPI

# Étape 1 : Builder
FROM python:3.9-slim as builder

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --user --no-cache-dir -r requirements.txt

# Étape 2 : Runtime
FROM python:3.9-slim

WORKDIR /app

# Copier les dépendances depuis le builder
COPY --from=builder /root/.local /root/.local

# Installer les dépendances système minimales
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ajouter .local/bin au PATH
ENV PATH=/root/.local/bin:$PATH

# Copier le code source
COPY api/ /app/api/
COPY src/ /app/src/
COPY models/ /app/models/
COPY mlruns/ /app/mlruns/

# Créer les dossiers nécessaires
RUN mkdir -p /app/logs /app/data

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]