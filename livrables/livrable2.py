from datasets import load_dataset
import matplotlib.pyplot as plt

# Charger le dataset
ds = load_dataset("Hemg/deepfake-and-real-images", split="train")

# Compter les classes
labels = [ds[i]["label"] for i in range(len(ds))]
num_real = labels.count(0)
num_fake = labels.count(1)

# Afficher stats
print(f"Nombre d'images réelles : {num_real}")
print(f"Nombre d'images deepfake : {num_fake}")

# Graphique
plt.bar(["Real", "Deepfake"], [num_real, num_fake], color=["green", "red"])
plt.title("Distribution des classes")
plt.ylabel("Nombre d'images")
plt.show()
