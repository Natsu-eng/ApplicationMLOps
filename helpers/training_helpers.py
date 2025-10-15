"""
Module de fonctions utilitaires pour l'entraînement ML.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from src.models.catalog import MODEL_CATALOG
from src.models.training import log_structured
from src.config.constants import TRAINING_CONSTANTS

class TrainingHelpers:
    """Helpers spécifiques à l'entraînement des modèles"""
    
    @staticmethod
    def get_task_specific_models(task_type: str) -> List[str]:
        """Retourne les modèles disponibles pour une tâche"""
        try:
            models = list(MODEL_CATALOG.get(task_type, {}).keys())
            return models
        except Exception as e:
            print(f"Erreur récupération modèles pour {task_type}: {str(e)[:100]}")
            log_structured("ERROR", f"Erreur récupération modèles pour {task_type}: {str(e)[:100]}")
            return []
    
    @staticmethod
    def get_default_models_for_task(task_type: str) -> List[str]:
        """Retourne les modèles par défaut pour une tâche"""
        default_models = {
            'classification': ['RandomForest', 'XGBoost', 'LogisticRegression'],
            'regression': ['RandomForest', 'XGBoost', 'LinearRegression'],
            'clustering': ['KMeans', 'DBSCAN', 'GaussianMixture']
        }
        available_models = TrainingHelpers.get_task_specific_models(task_type)
        models = [model for model in default_models.get(task_type, []) if model in available_models]
        return models
    
    @staticmethod
    def process_training_results(results: List[Dict], task_type: str) -> Dict[str, Any]:
        """Traite et analyse les résultats d'entraînement de façon robuste"""
        analysis = {
            "successful_models": [],
            "failed_models": [],
            "best_model": None,
            "performance_summary": {},
            "warnings": [],
            "recommendations": []
        }
        
        try:
            # Séparation modèles réussis/échoués
            for result in results:
                if result.get('success', False) and not result.get('metrics', {}).get('error'):
                    analysis["successful_models"].append(result)
                else:
                    analysis["failed_models"].append(result)
            
            # Analyse des modèles réussis
            if analysis["successful_models"]:
                # Détermination métrique principale
                primary_metric = (
                    'silhouette_score' if task_type == 'clustering' 
                    else 'r2' if task_type == 'regression' 
                    else 'accuracy'
                )
                
                # Recherche meilleur modèle
                valid_models = [
                    m for m in analysis["successful_models"] 
                    if m.get('metrics', {}).get(primary_metric) is not None
                ]
                
                if valid_models:
                    analysis["best_model"] = max(
                        valid_models, 
                        key=lambda x: x['metrics'][primary_metric]
                    )
                
                # Statistiques de performance
                metrics_data = {}
                for model in analysis["successful_models"]:
                    for metric, value in model.get('metrics', {}).items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            if metric not in metrics_data:
                                metrics_data[metric] = []
                            metrics_data[metric].append(value)
                
                analysis["performance_summary"] = {
                    metric: {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
                    for metric, values in metrics_data.items()
                }
                
                # Recommandations
                if len(analysis["successful_models"]) == 0:
                    analysis["recommendations"].append("Aucun modèle n'a réussi - vérifiez les données")
                elif len(analysis["successful_models"]) < len(results) / 2:
                    analysis["recommendations"].append("Moins de la moitié des modèles ont réussi - optimisez la configuration")
                
                # Warnings spécifiques
                if analysis["performance_summary"].get(primary_metric, {}).get('std', 0) > 0.1:
                    analysis["warnings"].append("Grande variance entre les modèles - données incohérentes")
            
            log_structured("INFO", "Analyse résultats entraînement", {
                "successful": len(analysis["successful_models"]),
                "failed": len(analysis["failed_models"]),
                "best_model": analysis["best_model"]["model_name"] if analysis["best_model"] else None
            })
            
        except Exception as e:
            log_structured("ERROR", f"Erreur analyse résultats: {str(e)[:100]}")
            analysis["warnings"].append(f"Erreur analyse: {str(e)[:100]}")
        
        return analysis
    
    @staticmethod
    def estimate_training_time(df: pd.DataFrame, n_models: int, task_type: str, optimize_hp: bool, n_features: int, use_smote: bool) -> int:
        """
        Estime le temps d'entraînement en secondes basé sur les paramètres fournis.

        Args:
            df (pd.DataFrame): DataFrame contenant les données.
            n_models (int): Nombre de modèles à entraîner.
            task_type (str): Type de tâche ('classification', 'regression', 'clustering').
            optimize_hp (bool): Si l'optimisation des hyperparamètres est activée.
            n_features (int): Nombre de features utilisées.
            use_smote (bool): Si SMOTE est utilisé (pour la classification).

        Returns:
            int: Temps estimé en secondes.
        """
        try:
            # Paramètres de base pour l'estimation
            base_time_per_model = TRAINING_CONSTANTS.get("BASE_TIME_PER_MODEL", 5)  # Temps de base par modèle (secondes)
            scaling_factor_rows = max(1, len(df) / 1000)  # Facteur basé sur le nombre de lignes
            scaling_factor_features = max(1, n_features / 10)  # Facteur basé sur le nombre de features
            hp_optimization_multiplier = 5 if optimize_hp else 1  # Multiplicateur pour l'optimisation des hyperparamètres
            smote_multiplier = 1.5 if use_smote and task_type == 'classification' else 1  # Multiplicateur pour SMOTE
            task_complexity = {
                'classification': 1.2,
                'regression': 1.0,
                'clustering': 1.5
            }.get(task_type, 1.0)  # Facteur de complexité par type de tâche

            # Calcul du temps estimé
            estimated_seconds = (
                base_time_per_model * 
                n_models * 
                scaling_factor_rows * 
                scaling_factor_features * 
                hp_optimization_multiplier * 
                smote_multiplier * 
                task_complexity
            )

            # Ajustement pour éviter des estimations trop faibles ou trop élevées
            estimated_seconds = max(10, min(estimated_seconds, 3600))  # Entre 10 secondes et 1 heure

            log_structured("INFO", "Estimation temps entraînement", {
                "n_models": n_models,
                "n_rows": len(df),
                "n_features": n_features,
                "task_type": task_type,
                "optimize_hp": optimize_hp,
                "use_smote": use_smote,
                "estimated_seconds": estimated_seconds
            })

            return int(estimated_seconds)

        except Exception as e:
            print(f"Erreur dans estimate_training_time: {str(e)[:100]}")
            log_structured("ERROR", f"Erreur estimation temps entraînement: {str(e)[:100]}")
            return 60  # Estimation par défaut de 1 minute