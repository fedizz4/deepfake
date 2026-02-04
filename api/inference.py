import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import io

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "model/model_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["REAL", "DEEPFAKE"]

# -------------------------------
# MODEL LOADING (UNE SEULE FOIS)
# -------------------------------
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------------------
# TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict(uploaded_file):
    """
    uploaded_file : UploadFile (FastAPI) ou bytes
    """
    if hasattr(uploaded_file, "file"):
        image_bytes = uploaded_file.file.read()
    else:
        image_bytes = uploaded_file

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = CLASS_NAMES[predicted.item()]
    confidence = round(confidence.item() * 100, 2)

    return {
        "status": label,
        "confidence": confidence
    }
