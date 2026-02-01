# src/preprocessing.py
"""
Pipeline de preprocessing pour FaceForensics++
"""
import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .dataset import FaceExtractor
import mlflow


class DataPreprocessor:
    def __init__(self, raw_data_dir, processed_data_dir):
        self.raw_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_data_dir)

        # Créer la structure
        for split in ['train', 'val', 'test']:
            for label in ['real', 'fake']:
                (self.processed_dir / split / label).mkdir(parents=True, exist_ok=True)

        # Logging MLflow
        mlflow.set_tracking_uri("../mlruns")
        mlflow.set_experiment("data-preprocessing")

    def organize_faceforensics(self, videos_per_class=50, frames_per_video=10):
        """
        Organise FaceForensics++ en images extraites

        Args:
            videos_per_class: Nombre de vidéos à traiter par classe
            frames_per_video: Nombre de frames à extraire par vidéo
        """
        print("🎬 Organisation de FaceForensics++...")

        # Initialiser l'extracteur
        extractor = FaceExtractor(device='cpu')

        all_images = []

        # 1. Extraire les vidéos réelles
        print("\n📹 Extraction des vidéos réelles...")
        real_videos = list((self.raw_dir / 'original_sequences' / 'youtube' / 'c23' / 'videos').glob('*.mp4'))

        for video_path in tqdm(real_videos[:videos_per_class]):
            faces = extractor.extract_from_video(
                video_path,
                self.processed_dir / 'raw_faces' / 'real',
                frames_per_video=frames_per_video
            )
            all_images.extend([(f, 'real') for f in faces])

        # 2. Extraire les vidéos fake (DeepFakes)
        print("\n🎭 Extraction des DeepFakes...")
        fake_videos = list((self.raw_dir / 'manipulated_sequences' / 'DeepFakes' / 'c23' / 'videos').glob('*.mp4'))

        for video_path in tqdm(fake_videos[:videos_per_class]):
            faces = extractor.extract_from_video(
                video_path,
                self.processed_dir / 'raw_faces' / 'fake',
                frames_per_video=frames_per_video
            )
            all_images.extend([(f, 'fake') for f in faces])

        print(f"\n✅ Extraction terminée: {len(all_images)} images")

        # 3. Split train/val/test
        self._split_dataset(all_images)

        # 4. Sauvegarder les métadonnées
        self._save_metadata(all_images)

        # Log MLflow
        with mlflow.start_run(run_name="data_preprocessing"):
            mlflow.log_params({
                "videos_per_class": videos_per_class,
                "frames_per_video": frames_per_video,
                "total_images": len(all_images)
            })

    def _split_dataset(self, all_images, train_ratio=0.7, val_ratio=0.15):
        """Split les données en train/val/test"""
        print("\n📊 Split des données...")

        # Séparer par classe
        real_images = [img for img, label in all_images if label == 'real']
        fake_images = [img for img, label in all_images if label == 'fake']

        # Split real
        real_train, real_temp = train_test_split(real_images, test_size=1 - train_ratio, random_state=42)
        real_val, real_test = train_test_split(real_temp, test_size=0.5, random_state=42)

        # Split fake
        fake_train, fake_temp = train_test_split(fake_images, test_size=1 - train_ratio, random_state=42)
        fake_val, fake_test = train_test_split(fake_temp, test_size=0.5, random_state=42)

        # Copier les fichiers
        splits = {
            'train': {'real': real_train, 'fake': fake_train},
            'val': {'real': real_val, 'fake': fake_val},
            'test': {'real': real_test, 'fake': fake_test}
        }

        for split_name, split_data in splits.items():
            for label, filepaths in split_data.items():
                for src_path in filepaths:
                    src = Path(src_path)
                    dst = self.processed_dir / split_name / label / src.name
                    shutil.copy2(src, dst)

        # Afficher les statistiques
        print(f"📈 Distribution finale:")
        for split_name in ['train', 'val', 'test']:
            real_count = len(list((self.processed_dir / split_name / 'real').glob('*.jpg')))
            fake_count = len(list((self.processed_dir / split_name / 'fake').glob('*.jpg')))
            print(f"  {split_name.upper()}: {real_count} réels, {fake_count} fakes")

    def _save_metadata(self, all_images):
        """Sauvegarde les métadonnées du dataset"""
        metadata = {
            "dataset": "FaceForensics++",
            "compression": "c23",
            "methods": ["original", "DeepFakes"],
            "total_images": len(all_images),
            "image_size": 224,
            "splits": {}
        }

        for split in ['train', 'val', 'test']:
            real_count = len(list((self.processed_dir / split / 'real').glob('*.jpg')))
            fake_count = len(list((self.processed_dir / split / 'fake').glob('*.jpg')))
            metadata["splits"][split] = {
                "real": real_count,
                "fake": fake_count,
                "total": real_count + fake_count
            }

        # Sauvegarder
        metadata_path = self.processed_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n💾 Métadonnées sauvegardées: {metadata_path}")

    def create_mock_data(self, n_images=100):
        """
        Crée des données mock pour tester le pipeline
        Utile si tu n'as pas encore téléchargé les vraies données
        """
        print("🎨 Création de données mock...")

        # Créer des images synthétiques
        for split in ['train', 'val', 'test']:
            for label in ['real', 'fake']:
                output_dir = self.processed_dir / split / label
                output_dir.mkdir(parents=True, exist_ok=True)

                n = n_images // 6  # Répartir équitablement

                for i in range(n):
                    # Créer une image synthétique
                    img = self._generate_mock_image(label)

                    # Sauvegarder
                    img_path = output_dir / f"{label}_{split}_{i:04d}.jpg"
                    cv2.imwrite(str(img_path), img)

        print(f"✅ Données mock créées: {n_images} images")

    def _generate_mock_image(self, label='real'):
        """Génère une image synthétique pour tests"""
        img = np.zeros((224, 224, 3), dtype=np.uint8)

        if label == 'real':
            # Fond bleu clair
            img[:, :] = [200, 220, 240]
            # Visage ovale
            cv2.ellipse(img, (112, 112), (80, 90), 0, 0, 360, [255, 220, 180], -1)
        else:
            # Fond rose clair
            img[:, :] = [240, 220, 240]
            # Visage avec artefacts
            cv2.ellipse(img, (112, 112), (80, 90), 0, 0, 360, [240, 200, 220], -1)
            # Rectangle d'artefact
            cv2.rectangle(img, (50, 50), (174, 174), [0, 255, 255], 2)

        return img


if __name__ == "__main__":
    # Exemple d'utilisation
    preprocessor = DataPreprocessor(
        raw_data_dir='../data/raw/faceforensicspp',
        processed_data_dir='../data/processed'
    )

    # Pour tester avec des données mock
    preprocessor.create_mock_data(n_images=100)

    # Pour les vraies données (une fois téléchargées)
    # preprocessor.organize_faceforensics(videos_per_class=10, frames_per_video=5)