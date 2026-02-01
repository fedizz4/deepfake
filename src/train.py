# src/train.py
"""
Script d'entraînement avec MLflow tracking
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from datetime import datetime
import json
from pathlib import Path

from .model import get_model, save_model
from .dataset import get_dataloaders
from .evaluate import calculate_metrics


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Setup MLflow
        mlflow.set_tracking_uri(config['mlflow_uri'])
        mlflow.set_experiment(config['experiment_name'])

        # Créer les dossiers de sauvegarde
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialiser le modèle
        self.model = get_model(
            model_name=config['model_name'],
            num_classes=1,
            pretrained=config.get('pretrained', True)
        ).to(self.device)

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

        # Early stopping
        self.patience = config.get('patience', 10)
        self.best_val_loss = float('inf')
        self.counter = 0

        print(f"✅ Trainer initialisé")
        print(f"   Modèle: {config['model_name']}")
        print(f"   Device: {self.device}")
        print(f"   Learning rate: {config['learning_rate']}")

    def train_epoch(self, dataloader):
        """Entraîne le modèle pour une epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze()

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item(): .4
            f})

            avg_loss = total_loss / len(dataloader)
            metrics = calculate_metrics(all_labels, all_preds)

            return avg_loss, metrics

        def validate(self, dataloader):
            """Valide le modèle"""
            self.model.eval()
            total_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Validation"):
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)

                    outputs = self.model(images).squeeze()
                    loss = self.criterion(outputs, labels)

                    total_loss += loss.item()
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_loss = total_loss / len(dataloader)
            metrics = calculate_metrics(all_labels, all_preds)

            return avg_loss, metrics

        def train(self, train_loader, val_loader, num_epochs):
            """Boucle d'entraînement principale"""

            with mlflow.start_run(run_name=self.config['run_name']):
                # Log des hyperparamètres
                mlflow.log_params(self.config)

                print(f"\n🚀 Début de l'entraînement pour {num_epochs} epochs")

                for epoch in range(num_epochs):
                    print(f"\n{'=' * 60}")
                    print(f"Epoch {epoch + 1}/{num_epochs}")
                    print(f"{'=' * 60}")

                    # Entraînement
                    train_loss, train_metrics = self.train_epoch(train_loader)
                    print(f"Train Loss: {train_loss:.4f} | AUC: {train_metrics['auc']:.4f}")

                    # Validation
                    val_loss, val_metrics = self.validate(val_loader)
                    print(f"Val Loss:   {val_loss:.4f} | AUC: {val_metrics['auc']:.4f}")

                    # Log MLflow
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("train_auc", train_metrics['auc'], step=epoch)
                    mlflow.log_metric("val_auc", val_metrics['auc'], step=epoch)
                    mlflow.log_metric("val_accuracy", val_metrics['accuracy'], step=epoch)
                    mlflow.log_metric("val_f1", val_metrics['f1'], step=epoch)

                    # Scheduler step
                    self.scheduler.step(val_loss)

                    # Early stopping check
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.counter = 0

                        # Sauvegarder le meilleur modèle
                        model_path = self.save_dir / f"best_model_{self.config['model_name']}.pth"
                        save_model(
                            self.model,
                            model_path,
                            metadata={
                                'epoch': epoch,
                                'val_loss': val_loss,
                                'val_auc': val_metrics['auc'],
                                'config': self.config
                            }
                        )
                        print(f"✅ Meilleur modèle sauvegardé: {model_path}")

                        # Log dans MLflow
                        mlflow.log_artifact(str(model_path))
                    else:
                        self.counter += 1
                        if self.counter >= self.patience:
                            print(f"⏹️  Early stopping à l'epoch {epoch + 1}")
                            break

                # Charger le meilleur modèle pour le retour
                best_model_path = self.save_dir / f"best_model_{self.config['model_name']}.pth"
                if best_model_path.exists():
                    checkpoint = torch.load(best_model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['state_dict'])

                return self.model

    def train_model(config_path='configs/training_config.yaml'):
        """Fonction principale d'entraînement"""

        # Charger la configuration
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Ajouter des métadonnées
        config['run_name'] = f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config['mlflow_uri'] = "../mlruns"
        config['save_dir'] = "models"

        # Créer les dataloaders
        datasets, dataloaders = get_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 2),
            max_samples=config.get('max_samples', None)
        )

        # Initialiser le trainer
        trainer = Trainer(config)

        # Entraîner
        model = trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            num_epochs=config['num_epochs']
        )

        # Évaluer sur le test set
        print(f"\n📊 Évaluation finale sur le test set...")
        test_loss, test_metrics = trainer.validate(dataloaders['test'])

        print(f"\n🎯 Résultats finaux:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test AUC:  {test_metrics['auc']:.4f}")
        print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Test F1:   {test_metrics['f1']:.4f}")

        # Sauvegarder les résultats
        results = {
            'test_metrics': test_metrics,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }

        results_path = Path("results") / f"results_{config['model_name']}.json"
        results_path.parent.mkdir(exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Résultats sauvegardés: {results_path}")

        return model, test_metrics

    if __name__ == "__main__":
        # Configuration par défaut
        default_config = {
            'experiment_name': 'deepfake-detection',
            'model_name': 'efficientnet_b0',
            'data_dir': '../data/processed',
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 20,
            'weight_decay': 1e-4,
            'patience': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'pretrained': True
        }

        # Sauvegarder la config par défaut
        config_dir = Path("configs")
        config_dir.mkdir(exist_ok=True)

        with open(config_dir / "training_config.yaml", 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        print("📄 Configuration par défaut créée: configs/training_config.yaml")

        # Démarrer l'entraînement
        model, metrics = train_model(config_dir / "training_config.yaml")