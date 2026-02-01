# api/inference.py
"""
Module d'inférence pour l'API
"""
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

from src.model import load_model
from src.dataset import get_transforms


class DeepfakeInference:
    """Classe pour gérer l'inférence des modèles"""

    def __init__(self, model_path: str = "models/best_model_efficientnet_b0.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.metadata = load_model(model_path, self.device)
        self.transform = get_transforms('val')

        print(f"✅ Inference engine initialisé")
        print(f"   Modèle: {self.metadata.get('backbone_name', 'unknown')}")
        print(f"   Device: {self.device}")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Prétraite une image pour l'inférence"""
        # Convertir en numpy array
        img_array = np.array(image.convert('RGB'))

        # Appliquer les transformations
        transformed = self.transform(image=img_array)
        img_tensor = transformed['image']

        return img_tensor.unsqueeze(0)  # Ajouter batch dimension

    def predict_single(self, image: Image.Image) -> Dict[str, Any]:
        """Prédire sur une seule image"""
        start_time = datetime.now()

        # Prétraiter
        img_tensor = self.preprocess_image(image).to(self.device)

        # Inférence
        with torch.no_grad():
            output = self.model(img_tensor).squeeze().cpu().numpy()

        # Post-process
        confidence = float(output)
        is_deepfake = confidence > 0.5

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "prediction": "FAKE" if is_deepfake else "REAL",
            "confidence": confidence,
            "is_deepfake": is_deepfake,
            "processing_time": processing_time,
            "threshold": 0.5
        }

    def predict_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Prédire sur un batch d'images"""
        results = []

        for idx, image in enumerate(images):
            try:
                result = self.predict_single(image)
                result["image_index"] = idx
                results.append(result)
            except Exception as e:
                results.append({
                    "image_index": idx,
                    "error": str(e),
                    "prediction": "ERROR",
                    "confidence": 0.0
                })

        return results

    def predict_from_path(self, image_path: str) -> Dict[str, Any]:
        """Prédire à partir d'un chemin de fichier"""
        try:
            image = Image.open(image_path)
            return self.predict_single(image)
        except Exception as e:
            return {
                "error": str(e),
                "prediction": "ERROR",
                "confidence": 0.0
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Obtenir les informations du modèle"""
        return {
            "backbone": self.metadata.get('backbone_name', 'unknown'),
            "device": str(self.device),
            "metadata": self.metadata.get('metadata', {}),
            "num_classes": self.metadata.get('num_classes', 1),
            "model_loaded": self.model is not None
        }

    def analyze_video_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyser une frame vidéo"""
        try:
            # Convertir numpy array en PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            return self.predict_single(image)
        except Exception as e:
            return {
                "error": str(e),
                "prediction": "ERROR",
                "confidence": 0.0
            }


class EnsembleInference:
    """Inférence avec un ensemble de modèles"""

    def __init__(self, model_paths: List[str]):
        self.models = []
        self.metadata_list = []

        for path in model_paths:
            model, metadata = load_model(path, 'cpu')
            self.models.append(model)
            self.metadata_list.append(metadata)

        self.transform = get_transforms('val')

        print(f"✅ Ensemble inference initialisé avec {len(self.models)} modèles")

    def predict_single(self, image: Image.Image) -> Dict[str, Any]:
        """Prédiction par vote d'ensemble"""
        predictions = []
        confidences = []

        for model in self.models:
            img_tensor = self.transform(image=np.array(image))['image'].unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor).squeeze().cpu().numpy()

            confidences.append(float(output))
            predictions.append(output > 0.5)

        # Moyenne des confidences
        avg_confidence = np.mean(confidences)

        # Vote majoritaire
        is_deepfake = sum(predictions) > len(predictions) / 2

        return {
            "prediction": "FAKE" if is_deepfake else "REAL",
            "confidence": avg_confidence,
            "individual_confidences": confidences,
            "vote_count": {
                "fake_votes": sum(predictions),
                "real_votes": len(predictions) - sum(predictions)
            },
            "is_deepfake": is_deepfake
        }


def test_inference():
    """Tester le module d'inférence"""
    print("🧪 Test du module d'inférence...")

    # Créer une image test synthétique
    img = Image.new('RGB', (224, 224), color='red')

    # Test inference simple
    inferencer = DeepfakeInference()

    # Prédiction
    result = inferencer.predict_single(img)

    print(f"\n📊 Résultat test:")
    print(f"  Prédiction: {result['prediction']}")
    print(f"  Confiance: {result['confidence']:.4f}")
    print(f"  Temps: {result['processing_time']:.4f}s")

    # Info modèle
    model_info = inferencer.get_model_info()
    print(f"\n📋 Info modèle:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_inference()