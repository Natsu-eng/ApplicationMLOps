"""
Détecteur unifié des types de tâches Computer Vision
Version finale compatible MVTec AD, classification, unsupervised, anomaly supervisée
"""

import numpy as np
from typing import Dict, Any, Tuple
from enum import Enum

class TaskType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    ANOMALY_DETECTION = "anomaly_detection"      # 0=normal, 1=anomaly (supervisé)
    UNSUPERVISED = "unsupervised"                # One-class (MVTec AD, train only)

def detect_cv_task(y: np.ndarray) -> Tuple[TaskType, Dict[str, Any]]:
    """
    Détecte la tâche À PARTIR DES LABELS DU TRAIN UNIQUEMENT !
    Validation robuste de la shape et du contenu de y
    """
    if y is None or len(y) == 0:
        return TaskType.UNSUPERVISED, {"n_classes": 0, "task": "empty"}

    # Validation shape: y doit être 1D
    if y.ndim != 1:
        raise ValueError(
            f"❌ ERREUR CRITIQUE: y a une shape incorrecte {y.shape}, "
            f"attendu 1D array (n_samples,). "
            f"Utilisez np.ravel() ou np.flatten() si nécessaire."
        )
    
    # ✅ Validation NaN/Inf
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError(
            f"❌ ERREUR CRITIQUE: y contient des valeurs NaN ou Inf. "
            f"Veuillez nettoyer les labels avant de détecter la tâche."
        )
    
    # Conversion en int si nécessaire
    if not np.issubdtype(y.dtype, np.integer):
        from src.shared.logging import get_logger
        logger = get_logger(__name__)
        logger.warning(f"⚠️ y n'est pas de type entier ({y.dtype}), conversion en int")
        y = y.astype(int)

    unique_labels = np.unique(y)
    n_classes = len(unique_labels)

    # CAS 1 : Uniquement des images normales → UNSUPERVISED (MVTec AD, CIFAR10-anomaly, etc.)
    if n_classes == 1:
        return TaskType.UNSUPERVISED, {
            "n_classes": 1,
            "task": "unsupervised_oneclass",
            "labels": unique_labels.tolist(),
            "description": "Anomaly Detection - Unsupervised (normal data only)"
        }

    # CAS 2 : Deux classes exactement {0,1} → Anomalie supervisée
    if n_classes == 2 and set(unique_labels.tolist()) == {0, 1}:
        return TaskType.ANOMALY_DETECTION, {
            "n_classes": 2,
            "task": "anomaly_detection",
            "is_binary": True,
            "labels": [0, 1],
            "description": "Anomaly Detection - Supervised (0=normal, 1=anomaly)"
        }

    # CAS 3 : Deux classes mais pas {0,1} → Classification binaire classique
    if n_classes == 2:
        return TaskType.BINARY_CLASSIFICATION, {
            "n_classes": 2,
            "task": "binary_classification",
            "is_binary": True,
            "labels": unique_labels.tolist()
        }

    # CAS 4 : Plus de 2 classes → Multiclass
    return TaskType.MULTICLASS_CLASSIFICATION, {
        "n_classes": n_classes,
        "task": "multiclass_classification",
        "is_binary": False,
        "labels": unique_labels.tolist()
    }