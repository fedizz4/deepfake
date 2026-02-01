# src/models/__init__.py
"""
Package des modèles
"""
from ..model import (
    DeepfakeDetector,
    EnsembleModel,
    get_model,
    save_model,
    load_model
)

__all__ = [
    'DeepfakeDetector',
    'EnsembleModel',
    'get_model',
    'save_model',
    'load_model'
]