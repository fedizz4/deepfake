# src/dataset.py
"""
Dataset PyTorch pour FaceForensics++
"""
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
import numpy as np
from pathlib import Path
import json


class FaceForensicsDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, max_samples=None):
        """
        Args:
            data_dir: Chemin vers data/processed/
            split: 'train', 'val', ou 'test'
            transform: Transformations à appliquer
            max_samples: Limite d'échantillons (pour debug)
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform

        # Charger les chemins des images
        self.real_images = list((self.data_dir / 'real').glob('*.jpg'))
        self.fake_images = list((self.data_dir / 'fake').glob('*.jpg'))

        # Limiter si nécessaire
        if max_samples:
            self.real_images = self.real_images[:max_samples // 2]
            self.fake_images = self.fake_images[:max_samples // 2]

        # Créer les labels
        self.images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)

        print(f"✅ Dataset {split}: {len(self.images)} images")
        print(f"   • Réels: {len(self.real_images)}, Fakes: {len(self.fake_images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Charger l'image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Appliquer les transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32),
            'path': str(img_path)
        }


def get_transforms(split='train'):
    """Retourne les transformations pour chaque split"""
    if split == 'train':
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:  # val/test
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


def get_dataloaders(data_dir, batch_size=32, num_workers=2, max_samples=None):
    """Crée les DataLoaders pour train, val, test"""

    datasets = {}
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        transform = get_transforms(split)
        dataset = FaceForensicsDataset(
            data_dir,
            split=split,
            transform=transform,
            max_samples=max_samples
        )

        datasets[split] = dataset

        # Sampler pour équilibrer les classes dans train
        if split == 'train':
            from torch.utils.data import WeightedRandomSampler
            # Calculer les poids pour chaque échantillon
            class_counts = [len(dataset.real_images), len(dataset.fake_images)]
            class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
            sample_weights = [class_weights[label] for label in dataset.labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False  # On utilise le sampler
        else:
            sampler = None
            shuffle = (split == 'train')

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )

        dataloaders[split] = dataloader

    return datasets, dataloaders


class FaceExtractor:
    """Classe pour extraire les visages des vidéos"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.mtcnn = MTCNN(
            keep_all=False,
            post_process=False,
            device=device,
            min_face_size=40
        )

    def extract_from_video(self, video_path, output_dir, frames_per_video=10):
        """Extrait les visages d'une vidéo"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // frames_per_video)

        extracted = []

        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                continue

            # Détecter le visage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(frame_rgb)

            if boxes is not None and len(boxes) > 0:
                # Prendre le visage avec la plus haute confiance
                x1, y1, x2, y2 = map(int, boxes[0])

                # Vérifier les dimensions
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue

                face = frame[y1:y2, x1:x2]

                # Sauvegarder
                face_filename = f"{Path(video_path).stem}_frame{i:04d}.jpg"
                face_path = output_dir / face_filename

                cv2.imwrite(str(face_path), cv2.resize(face, (224, 224)))
                extracted.append(str(face_path))

                if len(extracted) >= frames_per_video:
                    break

        cap.release()
        return extracted


if __name__ == "__main__":
    # Test du dataset
    datasets, dataloaders = get_dataloaders('../data/processed', batch_size=4, max_samples=100)

    for batch in dataloaders['train']:
        print(f"Batch shape: {batch['image'].shape}")
        print(f"Labels: {batch['label']}")
        break