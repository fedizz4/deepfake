# api/main.py
"""
API FastAPI pour la détection de deepfakes
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
from PIL import Image
import io
import numpy as np
import cv2
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Ajouter src au path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import load_model
from src.dataset import get_transforms
import albumentations as A

# Configuration
MODEL_PATH = Path("models/best_model_efficientnet_b0.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Charger le modèle
print("🔄 Chargement du modèle...")
try:
    model, metadata = load_model(MODEL_PATH, DEVICE)
    print(f"✅ Modèle chargé: {metadata.get('backbone_name', 'unknown')}")
    print(f"   Métadonnées: {json.dumps(metadata.get('metadata', {}), indent=2)}")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    model = None

# Initialiser FastAPI
app = FastAPI(
    title="Deepfake Detection API",
    description="API pour détecter les deepfakes dans les images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modèles Pydantic
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    is_deepfake: bool
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    predictions: list
    total_images: int
    processing_time: float


# Routes
@app.get("/", tags=["Root"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Deepfake Detection API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Vérifier l'état de l'API et du modèle"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(DEVICE),
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(image: UploadFile = File(...)):
    """
    Prédire si une image contient un deepfake

    Args:
        image: Image à analyser (JPEG/PNG)

    Returns:
        PredictionResponse: Résultat de la prédiction
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    start_time = datetime.now()

    try:
        # Lire et traiter l'image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        # Convertir en numpy array
        img_array = np.array(img)

        # Appliquer les transformations
        transform = get_transforms('val')  # Pas d'augmentation pour inference
        transformed = transform(image=img_array)
        img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

        # Prédiction
        with torch.no_grad():
            output = model(img_tensor).squeeze().cpu().numpy()

        # Interpréter les résultats
        confidence = float(output)
        is_deepfake = confidence > 0.5

        if is_deepfake:
            prediction = "FAKE"
        else:
            prediction = "REAL"

        # Temps de traitement
        processing_time = (datetime.now() - start_time).total_seconds()

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            is_deepfake=is_deepfake,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur traitement image: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(images: list[UploadFile] = File(...)):
    """
    Prédire un batch d'images

    Args:
        images: Liste d'images à analyser

    Returns:
        BatchPredictionResponse: Résultats batch
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    start_time = datetime.now()
    predictions = []

    for image_file in images:
        try:
            # Lire et traiter chaque image
            contents = await image_file.read()
            img = Image.open(io.BytesIO(contents)).convert('RGB')
            img_array = np.array(img)

            # Appliquer les transformations
            transform = get_transforms('val')
            transformed = transform(image=img_array)
            img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

            # Prédiction
            with torch.no_grad():
                output = model(img_tensor).squeeze().cpu().numpy()

            confidence = float(output)
            is_deepfake = confidence > 0.5

            predictions.append({
                "filename": image_file.filename,
                "prediction": "FAKE" if is_deepfake else "REAL",
                "confidence": confidence,
                "is_deepfake": is_deepfake
            })

        except Exception as e:
            predictions.append({
                "filename": image_file.filename,
                "error": str(e)
            })

    processing_time = (datetime.now() - start_time).total_seconds()

    return BatchPredictionResponse(
        predictions=predictions,
        total_images=len(images),
        processing_time=processing_time
    )


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Obtenir les informations du modèle chargé"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    return {
        "model_path": str(MODEL_PATH),
        "device": str(DEVICE),
        "metadata": metadata.get('metadata', {}),
        "backbone": metadata.get('backbone_name', 'unknown')
    }


@app.get("/logs/latest", tags=["Logs"])
async def get_latest_logs():
    """Obtenir les logs les plus récents"""
    log_file = Path("logs/api.log")
    if log_file.exists():
        return FileResponse(log_file)
    else:
        return {"message": "Aucun log disponible"}


if __name__ == "__main__":
    # Créer le dossier logs
    Path("logs").mkdir(exist_ok=True)

    # Démarrer le serveur
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )