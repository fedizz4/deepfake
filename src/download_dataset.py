from datasets import load_dataset

# Télécharge le dataset (split train par exemple)
ds = load_dataset("Hemg/deepfake-and-real-images", split="train")

print("Dataset téléchargé !")
print(f"Nombre d'images : {len(ds)}")
