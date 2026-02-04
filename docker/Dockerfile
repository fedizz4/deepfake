
# Image officielle PyTorch (Légère et stable)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# Installer dépendances système légères
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dépendances Python (hors torch)
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY api/ /app/api/
COPY src/ /app/src/

# Créer dossiers attendus
RUN mkdir -p /app/models /app/mlruns /app/logs /app/data

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
