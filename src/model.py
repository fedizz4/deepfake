# src/model.py
"""
Architectures de modèles pour la détection de deepfakes
"""
import torch
import torch.nn as nn
import torchvision.models as models
from timm import create_model
import mlflow


class DeepfakeDetector(nn.Module):
    """Modèle de détection de deepfakes avec différentes backbones"""

    def __init__(self, backbone='efficientnet_b0', num_classes=1, pretrained=True):
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes

        # Sélection du backbone
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'xception':
            self.backbone = create_model('xception', pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'vit_small':
            self.backbone = create_model('vit_small_patch16_224', pretrained=pretrained)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        else:
            raise ValueError(f"Backbone {backbone} non supporté")

        # Head de classification
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # Initialisation
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialise les poids du classifier"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)

        if self.backbone_name == 'vit_small':
            # ViT retourne (batch, seq_len, features)
            features = features[:, 0]  # Prendre le token [CLS]

        output = self.classifier(features)
        return torch.sigmoid(output) if self.num_classes == 1 else output

    def freeze_backbone(self):
        """Gèle les couches du backbone pour fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Dégèle les couches du backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class EnsembleModel(nn.Module):
    """Modèle d'ensemble combinant plusieurs backbones"""

    def __init__(self, backbones=['efficientnet_b0', 'resnet50'], num_classes=1):
        super().__init__()

        self.models = nn.ModuleList([
            DeepfakeDetector(backbone=bb, num_classes=num_classes)
            for bb in backbones
        ])

        # Fusion layer
        total_features = len(backbones) * 128  # Chaque modèle a 128 features avant output
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = []
        for model in self.models:
            # Extraire les features avant la dernière couche
            feat = model.backbone(x)
            if model.backbone_name == 'vit_small':
                feat = feat[:, 0]

            # Passer par toutes les couches sauf la dernière
            for layer in model.classifier[:-1]:
                feat = layer(feat)

            features.append(feat)

        # Concaténer les features
        combined = torch.cat(features, dim=1)
        output = self.fusion(combined)
        return torch.sigmoid(output) if self.num_classes == 1 else output


def get_model(model_name='efficientnet_b0', **kwargs):
    """Factory function pour créer des modèles"""
    model_factory = {
        'efficientnet_b0': lambda: DeepfakeDetector('efficientnet_b0', **kwargs),
        'resnet50': lambda: DeepfakeDetector('resnet50', **kwargs),
        'xception': lambda: DeepfakeDetector('xception', **kwargs),
        'vit_small': lambda: DeepfakeDetector('vit_small', **kwargs),
        'ensemble': lambda: EnsembleModel(['efficientnet_b0', 'resnet50'], **kwargs)
    }

    if model_name not in model_factory:
        raise ValueError(f"Modèle {model_name} non supporté. Choisir parmi: {list(model_factory.keys())}")

    return model_factory[model_name]()


def save_model(model, path, metadata=None):
    """Sauvegarde un modèle avec métadonnées"""
    checkpoint = {
        'state_dict': model.state_dict(),
        'backbone_name': model.backbone_name if hasattr(model, 'backbone_name') else 'ensemble',
        'num_classes': model.num_classes,
        'metadata': metadata or {}
    }

    torch.save(checkpoint, path)
    mlflow.pytorch.log_model(model, "model")


def load_model(path, device='cpu'):
    """Charge un modèle sauvegardé"""
    checkpoint = torch.load(path, map_location=device)

    if 'backbone_name' in checkpoint:
        model = get_model(
            checkpoint['backbone_name'],
            num_classes=checkpoint['num_classes']
        )
    else:
        model = get_model('efficientnet_b0')

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint.get('metadata', {})


if __name__ == "__main__":
    # Test des modèles
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test chaque architecture
    backbones = ['efficientnet_b0', 'resnet50', 'xception', 'vit_small']

    for backbone in backbones:
        print(f"\n🔧 Test {backbone}...")
        model = get_model(backbone, num_classes=1)
        model.to(device)

        # Test forward pass
        x = torch.randn(2, 3, 224, 224).to(device)
        output = model(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output values: {output[:2]}")

        # Compter les paramètres
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")