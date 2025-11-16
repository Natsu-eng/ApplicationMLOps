"""
Module de fonctions utilitaires pour l'entraînement ML.
✅ Déplacé depuis utils/training_helpers.py
✅ Fonctions complètes et production-ready
Version: 2.0
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.models.catalog import MODEL_CATALOG
from src.config.constants import TRAINING_CONSTANTS
from src.shared.logging import get_logger

logger = get_logger(__name__)


class TrainingHelpers:
    """Helpers spécifiques à l'entraînement des modèles"""
    
    @staticmethod
    def get_task_specific_models(task_type: str) -> List[str]:
        """Retourne les modèles disponibles pour une tâche"""
        try:
            models = list(MODEL_CATALOG.get(task_type, {}).keys())
            logger.info(f"✅ {len(models)} modèles disponibles pour {task_type}")
            return models
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles pour {task_type}: {e}")
            return []
    
    @staticmethod
    def get_default_models_for_task(task_type: str) -> List[str]:
        """Retourne les modèles par défaut recommandés pour une tâche"""
        default_models = {
            'classification': ['RandomForest', 'XGBoost', 'LogisticRegression', 'SVM'],
            'regression': ['RandomForest', 'XGBoost', 'LinearRegression', 'Ridge'],
            'clustering': ['KMeans', 'DBSCAN', 'GaussianMixture', 'AgglomerativeClustering']
        }
        
        available_models = TrainingHelpers.get_task_specific_models(task_type)
        recommended = [model for model in default_models.get(task_type, []) 
                      if model in available_models]
        
        logger.info(f"✅ {len(recommended)} modèles recommandés pour {task_type}")
        return recommended
    
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
                
                # Recommandations intelligentes
                analysis["recommendations"] = TrainingHelpers._generate_recommendations(
                    analysis, primary_metric, len(results)
                )
                
                # Warnings spécifiques
                if analysis["performance_summary"].get(primary_metric, {}).get('std', 0) > 0.1:
                    analysis["warnings"].append("Grande variance entre les modèles - données possiblement incohérentes")
            
            logger.info(f"✅ Analyse résultats: {len(analysis['successful_models'])} réussis, "
                       f"{len(analysis['failed_models'])} échoués")
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse résultats: {e}")
            analysis["warnings"].append(f"Erreur analyse: {str(e)[:100]}")
        
        return analysis
    
    @staticmethod
    def _generate_recommendations(analysis: Dict, primary_metric: str, total_models: int) -> List[str]:
        """Génère des recommandations intelligentes basées sur les résultats"""
        recommendations = []
        
        n_successful = len(analysis["successful_models"])
        
        # Taux de succès
        if n_successful == 0:
            recommendations.append("❌ Aucun modèle réussi - Vérifiez la qualité des données et la configuration")
        elif n_successful < total_models / 2:
            recommendations.append("⚠️ Moins de 50% de réussite - Optimisez le preprocessing et les hyperparamètres")
        elif n_successful == total_models:
            recommendations.append("✅ Tous les modèles ont réussi - Excellente configuration!")
        
        # Performance
        if analysis["best_model"]:
            best_score = analysis["best_model"]["metrics"].get(primary_metric, 0)
            
            if primary_metric == 'accuracy' and best_score < 0.7:
                recommendations.append("📊 Score accuracy faible (<70%) - Envisagez plus de données ou feature engineering")
            elif primary_metric == 'r2' and best_score < 0.5:
                recommendations.append("📈 Score R² faible (<0.5) - Essayez des modèles non-linéaires ou plus de features")
            elif primary_metric == 'silhouette_score' and best_score < 0.3:
                recommendations.append("🔍 Score silhouette faible (<0.3) - Testez différents nombres de clusters")
        
        # Variance
        if analysis["performance_summary"].get(primary_metric, {}).get('std', 0) > 0.15:
            recommendations.append("⚖️ Forte variance entre modèles - Données peut-être instables ou besoin de validation croisée")
        
        return recommendations
    
    @staticmethod
    def estimate_training_time(
        df: pd.DataFrame, 
        n_models: int, 
        task_type: str, 
        optimize_hp: bool, 
        n_features: int, 
        use_smote: bool
    ) -> int:
        """
        Estime le temps d'entraînement en secondes avec algorithme intelligent.
        
        Args:
            df: DataFrame des données
            n_models: Nombre de modèles à entraîner
            task_type: Type de tâche
            optimize_hp: Optimisation des hyperparamètres
            n_features: Nombre de features
            use_smote: Utilisation de SMOTE
            
        Returns:
            Temps estimé en secondes
        """
        try:
            # Paramètres de base
            base_time_per_model = TRAINING_CONSTANTS.get("BASE_TIME_PER_MODEL", 5)
            
            # Facteurs d'échelle
            scaling_factor_rows = max(1, len(df) / 1000)
            scaling_factor_features = max(1, n_features / 10)
            
            # Multiplicateurs
            hp_optimization_multiplier = 5 if optimize_hp else 1
            smote_multiplier = 1.5 if use_smote and task_type == 'classification' else 1
            
            # Complexité par tâche
            task_complexity = {
                'classification': 1.2,
                'regression': 1.0,
                'clustering': 1.5
            }.get(task_type, 1.0)
            
            # Calcul
            estimated_seconds = (
                base_time_per_model * 
                n_models * 
                scaling_factor_rows * 
                scaling_factor_features * 
                hp_optimization_multiplier * 
                smote_multiplier * 
                task_complexity
            )
            
            # Contraintes (entre 10s et 1h)
            estimated_seconds = max(10, min(estimated_seconds, 3600))
            
            logger.info(f"⏱️ Temps estimé: {estimated_seconds:.0f}s pour {n_models} modèles")
            
            return int(estimated_seconds)
            
        except Exception as e:
            logger.error(f"❌ Erreur estimation temps: {e}")
            return 60  # Fallback: 1 minute
    
    @staticmethod
    def format_training_time(seconds: int) -> str:
        """Formate le temps d'entraînement en format lisible"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}min {secs}s" if secs > 0 else f"{minutes}min"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h{minutes:02d}"
    
    @staticmethod
    def validate_model_selection(
        selected_models: List[str], 
        task_type: str,
        min_models: int = 1,
        max_models: int = 10
    ) -> Dict[str, Any]:
        """Valide la sélection de modèles"""
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Vérification nombre de modèles
        n_models = len(selected_models)
        
        if n_models < min_models:
            validation["is_valid"] = False
            validation["issues"].append(f"Sélectionnez au moins {min_models} modèle(s)")
        
        if n_models > max_models:
            validation["warnings"].append(f"Nombre élevé de modèles ({n_models}) - Temps d'entraînement long")
        
        # Vérification disponibilité
        available_models = TrainingHelpers.get_task_specific_models(task_type)
        invalid_models = [m for m in selected_models if m not in available_models]
        
        if invalid_models:
            validation["is_valid"] = False
            validation["issues"].append(f"Modèles invalides: {', '.join(invalid_models)}")
        
        # Recommandations
        if n_models == 1:
            validation["warnings"].append("Un seul modèle sélectionné - Impossible de comparer les performances")
        
        logger.info(f"✅ Validation sélection modèles: {validation['is_valid']}, "
                   f"{n_models} modèles pour {task_type}")
        
        return validation
    
    @staticmethod
    def get_model_complexity_info(model_name: str, task_type: str) -> Dict[str, str]:
        """Retourne les informations de complexité d'un modèle"""
        try:
            model_catalog = MODEL_CATALOG.get(task_type, {})
            model_config = model_catalog.get(model_name, {})
            
            complexity = model_config.get('complexity', 'medium')
            training_speed = model_config.get('training_speed', 'medium')
            
            complexity_labels = {
                'low': 'Débutant',
                'medium': 'Intermédiaire',
                'high': 'Expert'
            }
            
            return {
                'complexity': complexity,
                'complexity_label': complexity_labels.get(complexity, 'Intermédiaire'),
                'training_speed': training_speed,
                'category': model_config.get('category', 'Autres')
            }
        except Exception as e:
            logger.error(f"❌ Erreur récupération info complexité {model_name}: {e}")
            return {
                'complexity': 'medium',
                'complexity_label': 'Intermédiaire',
                'training_speed': 'medium',
                'category': 'Autres'
            }


# ============================================================================
# FONCTIONS UTILITAIRES STANDALONE
# ============================================================================

def filter_models_by_criteria(
    available_models: Dict[str, Dict],
    complexity_filter: List[str],
    speed_filter: str
) -> Dict[str, Dict]:
    """
    Filtre les modèles selon des critères de complexité et vitesse.
    
    Args:
        available_models: Dictionnaire des modèles disponibles
        complexity_filter: Liste des niveaux de complexité acceptés
        speed_filter: Filtre de vitesse ('Toutes', 'Rapide', 'Moyenne', 'Lente')
        
    Returns:
        Dictionnaire filtré des modèles
    """
    filtered = {}
    
    complexity_map = {
        'Débutant': 'low',
        'Intermédiaire': 'medium', 
        'Expert': 'high'
    }
    
    target_complexities = [complexity_map.get(c, 'medium') for c in complexity_filter]
    
    for name, config in available_models.items():
        # Filtre complexité
        model_complexity = config.get('complexity', 'medium')
        if model_complexity not in target_complexities:
            continue
        
        # Filtre vitesse
        if speed_filter != "Toutes":
            model_speed = config.get('training_speed', 'medium')
            if model_speed != speed_filter.lower():
                continue
        
        filtered[name] = config
    
    logger.info(f"✅ Filtrage modèles: {len(filtered)}/{len(available_models)} modèles retenus")
    return filtered


def categorize_models(models: Dict[str, Dict]) -> Dict[str, List[tuple]]:
    """
    Organise les modèles par catégorie.
    
    Args:
        models: Dictionnaire des modèles
        
    Returns:
        Dictionnaire avec catégories comme clés et listes de (nom, config) comme valeurs
    """
    categories = {}
    
    for model_name, config in models.items():
        category = config.get('category', '🧠 Autres')
        
        if category not in categories:
            categories[category] = []
        
        categories[category].append((model_name, config))
    
    # Tri alphabétique dans chaque catégorie
    for category in categories:
        categories[category].sort(key=lambda x: x[0])
    
    logger.info(f"✅ Modèles organisés en {len(categories)} catégories")
    return categories


def get_recommended_models(
    task_type: str,
    dataset_size: int,
    n_features: int,
    has_imbalance: bool = False
) -> List[str]:
    """
    Recommande des modèles basés sur les caractéristiques du dataset.
    
    Args:
        task_type: Type de tâche
        dataset_size: Nombre d'échantillons
        n_features: Nombre de features
        has_imbalance: Présence de déséquilibre (classification)
        
    Returns:
        Liste de noms de modèles recommandés
    """
    recommendations = []
    
    if task_type == 'classification':
        # Petit dataset
        if dataset_size < 1000:
            recommendations = ['LogisticRegression', 'SVM', 'KNN']
        # Dataset moyen
        elif dataset_size < 10000:
            recommendations = ['RandomForest', 'XGBoost', 'SVM']
        # Grand dataset
        else:
            recommendations = ['XGBoost', 'LightGBM', 'RandomForest']
        
        # Ajustement pour déséquilibre
        if has_imbalance:
            # Privilégier les modèles robustes au déséquilibre
            if 'RandomForest' not in recommendations:
                recommendations.insert(0, 'RandomForest')
            if 'XGBoost' not in recommendations:
                recommendations.insert(0, 'XGBoost')
    
    elif task_type == 'regression':
        if dataset_size < 1000:
            recommendations = ['LinearRegression', 'Ridge', 'Lasso']
        elif dataset_size < 10000:
            recommendations = ['RandomForest', 'GradientBoosting', 'Ridge']
        else:
            recommendations = ['XGBoost', 'LightGBM', 'RandomForest']
    
    elif task_type == 'clustering':
        if n_features > 10:
            recommendations = ['KMeans', 'DBSCAN', 'SpectralClustering']
        else:
            recommendations = ['KMeans', 'GaussianMixture', 'AgglomerativeClustering']
    
    # Vérifier disponibilité
    available = TrainingHelpers.get_task_specific_models(task_type)
    recommendations = [m for m in recommendations if m in available]
    
    logger.info(f"✅ {len(recommendations)} modèles recommandés pour {task_type} "
               f"(dataset: {dataset_size}, features: {n_features})")
    
    return recommendations[:5]  # Max 5 recommandations


# Export
__all__ = [
    'TrainingHelpers',
    'filter_models_by_criteria',
    'categorize_models',
    'get_recommended_models'
]