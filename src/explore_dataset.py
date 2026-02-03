from datasets import load_dataset
import matplotlib.pyplot as plt

# Téléchargement du dataset
ds = load_dataset("Hemg/deepfake-and-real-images", split="train")

print(ds)

# Afficher quelques exemples
for i in range(5):
    image = ds[i]["image"]
    label = ds[i]["label"]
    plt.imshow(image)
    plt.title("Deepfake" if label == 1 else "Real")
    plt.axis('off')
    plt.show()

# Statistiques simples
labels = [ds[i]["label"] for i in range(len(ds))]
num_real = labels.count(0)
num_fake = labels.count(1)
print(f"Nombre d'images réelles : {num_real}")
print(f"Nombre d'images deepfake : {num_fake}")
