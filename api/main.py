from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .inference import predict
import io

app = FastAPI(title="Deepfake Image Detector")

# 1. Ajout du Middleware CORS (C'est ce qui règle l'erreur 403)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dans un vrai projet, remplace "*" par l'URL de ton frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # 2. Lire le contenu du fichier uploadé
    image_bytes = await file.read()

    # 3. Envoyer les bytes à ta fonction d'inférence
    # Assure-toi que ta fonction predict dans inference.py accepte des bytes ou un flux
    result = predict(image_bytes)

    return result