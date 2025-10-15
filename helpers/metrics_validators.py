"""
Validation des données pour les calculs de métriques ML.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
from helpers.data_transformers import safe_array_conversion

logger = logging.getLogger(__name__)

def validate_input_data(y_true: Any, y_pred: Any, task_type: str) -> Dict[str, Any]:
    """
    Valide les données d'entrée de façon exhaustive pour les métriques.
    """
    validation = {
        "is_valid": False,
        "issues": [],
        "warnings": [],
        "n_samples": 0,
        "task_type": task_type.lower().strip()
    }
    
    try:
        # Normalisation task_type
        if validation["task_type"] in ['unsupervised', 'cluster']:
            validation["task_type"] = 'clustering'
        
        # Conversion sécurisée
        y_true_flat = safe_array_conversion(y_true, sample=False)
        y_pred_flat = safe_array_conversion(y_pred, sample=False)
        
        # Validation de base
        if len(y_true_flat) == 0 or len(y_pred_flat) == 0:
            validation["issues"].append("Données vides après conversion")
            return validation
        
        if len(y_true_flat) != len(y_pred_flat):
            validation["issues"].append(
                f"Dimensions incohérentes: y_true={len(y_true_flat)}, y_pred={len(y_pred_flat)}"
            )
            return validation
        
        validation["n_samples"] = len(y_true_flat)
        
        # Validation spécifique au task_type
        if validation["task_type"] == "classification":
            _validate_classification_data(y_true_flat, y_pred_flat, validation)
        elif validation["task_type"] == "regression":
            _validate_regression_data(y_true_flat, y_pred_flat, validation)
        elif validation["task_type"] == "clustering":
            _validate_clustering_data(y_true_flat, y_pred_flat, validation)
        else:
            validation["issues"].append(f"Type de tâche non supporté: {validation['task_type']}")
            return validation
        
        # Validation de taille
        min_samples = 2  # Minimum pour les calculs
        if validation["n_samples"] < min_samples:
            validation["warnings"].append(
                f"Peu d'échantillons: {validation['n_samples']} < {min_samples}"
            )
        
        # Si pas d'issues critiques, validation réussie
        if not validation["issues"]:
            validation["is_valid"] = True
        
        logger.debug(f"Validation données terminée: {validation['n_samples']} échantillons")
        
    except Exception as e:
        logger.error(f"Erreur critique validation données: {e}")
        validation["issues"].append(f"Erreur validation: {str(e)}")
    
    return validation

def _validate_classification_data(y_true: np.ndarray, y_pred: np.ndarray, validation: Dict):
    """Validation spécifique classification."""
    try:
        unique_true = np.unique(y_true[~np.isnan(y_true)])
        unique_pred = np.unique(y_pred[~np.isnan(y_pred)])
        
        if len(unique_true) < 2:
            validation["issues"].append("Moins de 2 classes dans y_true")
        
        if len(unique_pred) < 2:
            validation["warnings"].append("Moins de 2 classes dans y_pred")
        
        max_classes = 50
        if len(unique_true) > max_classes:
            validation["warnings"].append(f"Trop de classes: {len(unique_true)} > {max_classes}")
            
    except Exception as e:
        validation["issues"].append(f"Erreur validation classification: {str(e)}")

def _validate_regression_data(y_true: np.ndarray, y_pred: np.ndarray, validation: Dict):
    """Validation spécifique régression."""
    try:
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            validation["warnings"].append("Valeurs infinies détectées")
        
        nan_count = np.sum(np.isnan(y_true)) + np.sum(np.isnan(y_pred))
        max_missing_ratio = 0.5
        
        if nan_count > len(y_true) * max_missing_ratio:
            validation["warnings"].append(f"Trop de NaN: {nan_count}/{len(y_true)} valeurs")
            
    except Exception as e:
        validation["issues"].append(f"Erreur validation régression: {str(e)}")

def _validate_clustering_data(y_true: np.ndarray, y_pred: np.ndarray, validation: Dict):
    """Validation spécifique clustering."""
    try:
        unique_labels = np.unique(y_pred[~np.isnan(y_pred)])
        
        if len(unique_labels) < 2 and -1 not in unique_labels:
            validation["warnings"].append("Moins de 2 clusters valides")
            
    except Exception as e:
        validation["issues"].append(f"Erreur validation clustering: {str(e)}")