import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models

from dataset import DeepfakeDataset  # Assure-toi que dataset.py est correct

# -------------------------------
# CONFIG
# -------------------------------
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 5  # augmente plus tard si tu veux
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Vérification Hardware ---")
print(f"Device utilisé : {DEVICE}")

if DEVICE.type == 'cuda':
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
    print(f"Mémoire totale : {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)} GB")
else:
    print("ATTENTION : Le GPU n'est pas détecté, l'entraînement sera très lent !")
MODEL_PATH = "models/deepfake_cnn.pth"

# -------------------------------
# DATASET
# -------------------------------
print("Chargement du dataset...")
full_dataset = DeepfakeDataset(split="train")
print(f"Dataset chargé avec {len(full_dataset)} images.")

# Split train / val 80/20
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# MODEL (ResNet18)
# -------------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes : real / deepfake
model = model.to(DEVICE)

# -------------------------------
# LOSS & OPTIMIZER
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# TRAIN LOOP
# -------------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

    # -------------------------------
    # VALIDATION
    # -------------------------------
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%\n")

# -------------------------------
# SAVE MODEL
# -------------------------------
torch.save(model.state_dict(), MODEL_PATH)
print(f"Modèle sauvegardé dans {MODEL_PATH}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Vérification Hardware ---")
print(f"Device utilisé : {DEVICE}")

if DEVICE.type == 'cuda':
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
    print(f"Mémoire totale : {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)} GB")
else:
    print("ATTENTION : Le GPU n'est pas détecté, l'entraînement sera très lent !")
