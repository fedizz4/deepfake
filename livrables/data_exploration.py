import os
from datasets import load_dataset
import matplotlib.pyplot as plt

# -------------------------------
# Charger le dataset
# -------------------------------
# Si tu as un HF_TOKEN, tu peux le définir ici :
# os.environ["HF_TOKEN"] = "ton_token_ici"

print("Chargement du dataset...")
ds = load_dataset("Hemg/deepfake-and-real-images", split="train")
print(f"Dataset chargé avec {len(ds)} images.")

# -------------------------------
# Statistiques de classes
# -------------------------------
labels = [ds[i]["label"] for i in range(len(ds))]
num_real = labels.count(0)
num_fake = labels.count(1)

print(f"Nombre d'images réelles : {num_real}")
print(f"Nombre d'images deepfake : {num_fake}")

# Graphique de distribution
plt.figure(figsize=(6,4))
plt.bar(["Real", "Deepfake"], [num_real, num_fake], color=["green", "red"])
plt.title("Distribution des classes")
plt.ylabel("Nombre d'images")
plt.show()

# -------------------------------
# Afficher quelques images
# -------------------------------
def show_examples(dataset, label_value, num_examples=5):
    """Affiche quelques images d'une classe spécifique"""
    count = 0
    plt.figure(figsize=(15,3))
    for item in dataset:
        if item["label"] == label_value:
            plt.subplot(1, num_examples, count+1)
            plt.imshow(item["image"])
            plt.title("Deepfake" if label_value==1 else "Real")
            plt.axis('off')
            count += 1
            if count >= num_examples:
                break
    plt.show()

# 5 images réelles
show_examples(ds, label_value=0, num_examples=5)

# 5 images deepfake
show_examples(ds, label_value=1, num_examples=5)
