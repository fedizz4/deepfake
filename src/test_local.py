import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- CONFIGURATION ---
MODEL_PATH = "models/deepfake_cnn.pth"
# Choisis une image que tu as sur ton PC pour tester
IMAGE_TO_TEST = "ma_photo_test.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Recréer l'architecture (ResNet18)
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 2. Préparation de l'image (doit être identique à l'entraînement)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict(img_path):
    if not os.path.exists(img_path):
        print(f"❌ Erreur : Le fichier {img_path} n'existe pas.")
        return

    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = "REAL" if predicted.item() == 0 else "DEEPFAKE"
    color = "✅" if label == "REAL" else "🚨"

    print(f"\n--- RÉSULTAT DU TEST ---")
    print(f"Statut : {color} {label}")
    print(f"Confiance : {confidence.item() * 100:.2f}%")


if __name__ == "__main__":
    predict(IMAGE_TO_TEST)