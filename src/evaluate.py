# src/evaluate.py
"""
Évaluation et visualisation des modèles
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
import mlflow
from pathlib import Path
import json
from tqdm import tqdm


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calcule toutes les métriques d'évaluation

    Args:
        y_true: Labels réels
        y_pred: Prédictions (probabilités)
        threshold: Seuil pour la classification binaire

    Returns:
        dict: Dictionnaire de métriques
    """
    # Convertir en array numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification binaire avec seuil
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calcul des métriques
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5,
        'threshold': threshold
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred, threshold=0.5, save_path=None):
    """Trace et sauvegarde la matrice de confusion"""
    y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return cm


def plot_roc_curve(y_true, y_pred, save_path=None):
    """Trace la courbe ROC"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)  # Ligne diagonale
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fpr, tpr, thresholds


def plot_predictions_distribution(y_true, y_pred, save_path=None):
    """Trace la distribution des prédictions par classe"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Séparer par classe
    real_preds = y_pred[y_true == 0]
    fake_preds = y_pred[y_true == 1]

    plt.figure(figsize=(10, 6))

    plt.hist(real_preds, bins=50, alpha=0.5, label='Real (label=0)',
             color='blue', density=True)
    plt.hist(fake_preds, bins=50, alpha=0.5, label='Fake (label=1)',
             color='red', density=True)

    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')

    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Distribution des Prédictions par Classe')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def evaluate_model(model, dataloader, device='cpu', threshold=0.5):
    """
    Évalue un modèle sur un dataloader

    Returns:
        dict: Métriques d'évaluation
        np.array: Prédictions
        np.array: Labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            paths = batch['path']

            outputs = model(images).squeeze()

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    # Calculer les métriques
    metrics = calculate_metrics(all_labels, all_preds, threshold)

    # Trouver les erreurs
    errors = find_misclassified(all_labels, all_preds, all_paths, threshold)

    return metrics, np.array(all_preds), np.array(all_labels), errors


def find_misclassified(y_true, y_pred, paths, threshold=0.5):
    """Trouve les images mal classées"""
    y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
    misclassified = []

    for i, (true, pred) in enumerate(zip(y_true, y_pred_binary)):
        if true != pred:
            misclassified.append({
                'path': paths[i],
                'true_label': 'real' if true == 0 else 'fake',
                'pred_label': 'real' if pred == 0 else 'fake',
                'confidence': float(y_pred[i])
            })

    return misclassified


def save_evaluation_results(metrics, predictions, labels, save_dir='results'):
    """Sauvegarde les résultats d'évaluation"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Sauvegarder les métriques
    metrics_path = save_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Sauvegarder les prédictions
    predictions_data = {
        'labels': labels.tolist(),
        'predictions': predictions.tolist(),
        'metrics': metrics
    }

    predictions_path = save_dir / 'predictions.json'
    with open(predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)

    # Générer des visualisations
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Matrice de confusion
    cm_path = plots_dir / 'confusion_matrix.png'
    plot_confusion_matrix(labels, predictions, save_path=cm_path)

    # Courbe ROC
    roc_path = plots_dir / 'roc_curve.png'
    plot_roc_curve(labels, predictions, save_path=roc_path)

    # Distribution des prédictions
    dist_path = plots_dir / 'predictions_distribution.png'
    plot_predictions_distribution(labels, predictions, save_path=dist_path)

    # Log dans MLflow
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(predictions_path))
        mlflow.log_artifact(str(plots_dir))

    print(f"✅ Résultats sauvegardés dans: {save_dir}")
    return save_dir


def compare_models(results_dict, save_path=None):
    """Compare plusieurs modèles avec un bar plot"""
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))

    for i, metric in enumerate(metrics):
        values = [results_dict[model].get(metric, 0) for model in models]

        axes[i].bar(models, values, color='skyblue')
        axes[i].set_title(f'{metric.upper()}')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim([0, 1])
        axes[i].tick_params(axis='x', rotation=45)

        # Ajouter les valeurs sur les bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')

    plt.suptitle('Comparaison des Modèles', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Exemple d'utilisation
    from src.dataset import get_dataloaders
    from src.model import get_model

    # Charger les données
    datasets, dataloaders = get_dataloaders(
        data_dir='../data/processed',
        batch_size=32,
        max_samples=100
    )

    # Charger un modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('efficientnet_b0').to(device)

    # Évaluer
    metrics, predictions, labels, errors = evaluate_model(
        model, dataloaders['test'], device
    )

    print("\n📊 Métriques d'évaluation:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Sauvegarder
    save_evaluation_results(metrics, predictions, labels)