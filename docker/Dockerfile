# Étape 1 : Builder (on garde une image légère pour compiler)
FROM python:3.9-slim as builder

WORKDIR /app

# Installer les dépendances système nécessaires à la compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Installation des dépendances Python (en s'assurant d'avoir torch avec CUDA)
# Note : Ton requirements.txt doit pointer vers une version compatible CUDA
RUN pip install --user --no-cache-dir -r requirements.txt

# Étape 2 : Runtime (Utilisation de l'image de base NVIDIA avec CUDA)
# On utilise l'image officielle NVIDIA compatible avec ton driver (CUDA 11.8)
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Installer Python et les libs système minimales dans l'image NVIDIA
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier les librairies installées depuis le builder
COPY --from=builder /root/.local /root/.local

# Configurer les variables d'environnement
ENV PATH=/root/.local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Force Python à utiliser l'encodage UTF-8
ENV PYTHONIOENCODING=utf-8

# Copier ton code et tes modèles
COPY api/ /app/api/
COPY src/ /app/src/
COPY models/ /app/models/
COPY mlruns/ /app/mlruns/

# Créer les dossiers nécessaires avec les bonnes permissions
RUN mkdir -p /app/logs /app/data && chmod -R 777 /app/logs /app/data

# Exposer le port de FastAPI
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]