from datasets import load_dataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DeepfakeDataset(Dataset):
    def __init__(self, split="train"):
        self.dataset = load_dataset(
            "Hemg/deepfake-and-real-images",
            split=split
        )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        image = self.transform(image)
        return image, label
