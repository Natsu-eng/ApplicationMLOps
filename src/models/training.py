"""
Module d'entraînement robuste pour le machine learning.
Supporte l'apprentissage supervisé et non-supervisé avec gestion MLOps avancée.
Version Production - 
"""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold
from imblearn.pipeline import Pipeline # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
import joblib
import os
import time
import gc
from typing import Dict, List, Any, Optional, Tuple
import warnings
from sklearn.exceptions import ConvergenceWarning

# Import des modules déplacés
from monitoring.performance_monitor import TrainingMonitor
from monitoring.state_managers import STATE
from monitoring.training_state_manager import TrainingStateManager, TRAINING_STATE
from monitoring.mlflow_collector import MLflowRunCollector
from helpers.data_validators import DataValidator

from src.evaluation.metrics import evaluate_single_train_test_split
from utils.mlflow import _ensure_array_like, _safe_cluster_metrics, clean_model_name, format_mlflow_run_for_ui, get_git_info, is_mlflow_available

# Initialisation du collecteur MLflow
from monitoring.mlflow_collector import get_mlflow_collector
MLFLOW_COLLECTOR = get_mlflow_collector()

# Intégration MLflow
try:
    import mlflow # type: ignore
    import mlflow.sklearn # type: ignore
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from joblib import Parallel, delayed

# Imports conditionnels pour robustesse
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    STREAMLIT_AVAILABLE = False

# Configuration des warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import des modules de l'application
try:
    from src.models.catalog import get_model_config
    from src.data.preprocessing import create_preprocessor, safe_label_encode, validate_preprocessor
    from src.evaluation.metrics import EvaluationMetrics
    from src.data.data_analysis import auto_detect_column_types
    from src.shared.logging import get_logger
    from src.config.constants import TRAINING_CONSTANTS, PREPROCESSING_CONSTANTS, LOGGING_CONSTANTS, MLFLOW_CONSTANTS
except ImportError as e:
    print(f"Warning: Some imports failed - {e}")
    # Fallback pour les tests
    def get_model_config(*args, **kwargs): return None
    def auto_detect_column_types(*args, **kwargs): return {}
    def get_logger(name): 
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)
    TRAINING_CONSTANTS = {}
    PREPROCESSING_CONSTANTS = {}

# Utilisation du système de logging centralisé
logger = get_logger(__name__)


# ============================================================================
# 🆕 CLASSE DE VALIDATION CENTRALISÉE
# ============================================================================

class FeatureListValidator:
    """
    Validateur centralisé pour feature_list.
    Garantit la cohérence entre tous les modules.
    """
    
    @staticmethod
    def validate_and_extract(
        preprocessing_choices: Dict[str, Any],
        column_types: Dict[str, List[str]],
        model_name: str
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Valide et extrait feature_list de manière robuste.  
        Returns:
            Tuple (feature_list validée, column_types filtrés)
        """
        # RÉCUPÉRATION de feature_list
        feature_list = preprocessing_choices.get('feature_list', [])
        
        if not feature_list:
            logger.error(f"❌ {model_name}: feature_list VIDE dans preprocessing_choices!")
            logger.error(f"   preprocessing_choices keys: {list(preprocessing_choices.keys())}")
            raise ValueError(f"{model_name}: feature_list manquante dans preprocessing_choices")
        
        # CONVERSION en set pour filtrage rapide
        feature_set = set(feature_list)
        
        logger.info(f"✅ {model_name}: {len(feature_list)} features validées")
        logger.debug(f"   Features: {feature_list[:10]}...")
        
        # FILTRAGE de column_types
        filtered_column_types = {}
        columns_removed = {}
        columns_kept = {}
        
        for col_type, cols in column_types.items():
            valid_cols = [col for col in cols if col in feature_set]
            removed_cols = [col for col in cols if col not in feature_set]
            
            if valid_cols:
                filtered_column_types[col_type] = valid_cols
                columns_kept[col_type] = len(valid_cols)
            
            if removed_cols:
                columns_removed[col_type] = removed_cols
        
        # LOGGING détaillé
        if columns_removed:
            logger.warning(f"⚠️ {model_name}: Colonnes RETIRÉES (absentes de feature_list):")
            for col_type, removed_cols in columns_removed.items():
                logger.warning(f"   • {col_type}: {len(removed_cols)} colonnes → {removed_cols[:5]}...")
        
        if columns_kept:
            logger.info(f"✅ {model_name}: Colonnes CONSERVÉES:")
            for col_type, count in columns_kept.items():
                logger.info(f"   • {col_type}: {count} colonnes")
        
        # VALIDATION finale
        total_kept = sum(len(cols) for cols in filtered_column_types.values())
        if total_kept == 0:
            logger.error(f"❌ {model_name}: AUCUNE colonne conservée après filtrage!")
            logger.error(f"   feature_list: {feature_list[:10]}")
            logger.error(f"   column_types original: {[(k, len(v)) for k, v in column_types.items()]}")
            raise ValueError(f"{model_name}: Aucune colonne valide après filtrage")
        
        logger.info(f"✅ {model_name}: {total_kept} colonnes au total après filtrage")
        
        return feature_list, filtered_column_types

# ============================================
# FONCTIONS DE PIPELINE AVEC GESTION D'ERREURS 
# ============================================

def create_leak_free_pipeline(
    model_name: str, 
    task_type: str, 
    column_types: Dict[str, List[str]],
    preprocessing_choices: Dict[str, Any],
    use_smote: bool = False,
    optimize_hyperparams: bool = False
) -> Tuple[Optional[Pipeline], Optional[Dict]]:
    """
    VERSION CORRIGÉE avec validation SMOTE atomique
    """
    try:
        logger.info(f"🔧 Création pipeline pour {model_name} (task: {task_type}, SMOTE: {use_smote})")
        
        # RÉCUPÉRATION configuration modèle
        model_config = get_model_config(task_type, model_name)
        if not model_config:
            logger.error(f"❌ Configuration non trouvée pour {model_name} ({task_type})")
            return None, None
        
        model = model_config["model"]
        
        # PRÉPARATION grille de paramètres
        param_grid = {}
        if optimize_hyperparams and "params" in model_config:
            param_grid = {f"model__{k}": v for k, v in model_config["params"].items()}
        
        # VALIDATION CENTRALISÉE via FeatureListValidator
        try:
            feature_list, filtered_column_types = FeatureListValidator.validate_and_extract(
                preprocessing_choices=preprocessing_choices,
                column_types=column_types,
                model_name=model_name
            )
        except ValueError as e:
            logger.error(f"❌ Validation feature_list échouée pour {model_name}: {e}")
            return None, None
        
        # 🆕 VALIDATION SMOTE AVANT CRÉATION PIPELINE
        final_use_smote = False
        if use_smote and task_type == 'classification':
            # Vérification k_neighbors depuis preprocessing_choices
            smote_k = preprocessing_choices.get("smote_k_neighbors", 5)
            
            # ⚠️ CRITIQUE : Vérifier que DataFrame est accessible
            # On ne peut pas valider ici sans accès aux données
            # => Délégation à train_single_model_with_mlflow
            logger.info(f"🔍 SMOTE demandé avec k={smote_k}, validation différée à l'entraînement")
            final_use_smote = True
        
        # CRÉATION du préprocesseur avec colonnes FILTRÉES
        preprocessor = create_preprocessor(preprocessing_choices, filtered_column_types)
        if preprocessor is None:
            logger.error(f"❌ Échec création préprocesseur pour {model_name}")
            return None, None
        
        # CONSTRUCTION du pipeline
        if final_use_smote:
            logger.info("🔄 Construction pipeline avec SMOTE")
            
            smote_k = preprocessing_choices.get("smote_k_neighbors", 5)
            random_state = preprocessing_choices.get("random_state", 42)
            sampling_strategy = preprocessing_choices.get("smote_sampling_strategy", 'auto')
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(
                    random_state=random_state,
                    k_neighbors=smote_k,
                    sampling_strategy=sampling_strategy
                )),
                ('model', model)
            ])
            
            logger.info(f"✅ Pipeline créé avec 3 étapes: preprocessor → SMOTE(k={smote_k}) → {model_name}")
        
        else:
            logger.info("🔄 Construction pipeline standard")
            
            if use_smote and task_type != 'classification':
                logger.warning(f"⚠️ SMOTE ignoré pour task_type='{task_type}'")
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            logger.info(f"✅ Pipeline créé avec 2 étapes: preprocessor → {model_name}")
        
        # VALIDATION finale du pipeline
        expected_steps = ['preprocessor', 'model'] if not final_use_smote else ['preprocessor', 'smote', 'model']
        actual_steps = list(pipeline.named_steps.keys())
        
        if actual_steps != expected_steps:
            logger.error(f"❌ Pipeline invalide! Attendu: {expected_steps}, Obtenu: {actual_steps}")
            return None, None
        
        for step_name in expected_steps:
            if pipeline.named_steps[step_name] is None:
                logger.error(f"❌ Étape '{step_name}' est None dans le pipeline!")
                return None, None
        
        logger.info(f"✅ Pipeline validé avec succès pour {model_name}")
        return pipeline, param_grid if param_grid else None
    
    except Exception as e:
        logger.error(f"❌ Erreur création pipeline pour {model_name}: {e}", exc_info=True)
        return None, None

# ===================================
# FONCTIONS D'ENTRAÎNEMENT PAR MODÈLE 
# ===================================

def train_single_model_supervised(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: Dict = None,
    task_type: str = 'classification',
    monitor: TrainingMonitor = None
) -> Dict[str, Any]:
    """
    VERSION ROBUSTE avec garantie metrics TOUJOURS présentes
    - Initialisation metrics par défaut AVANT try/except
    - Calcul immédiat après entraînement
    - Fallback metrics en cas d'erreur évaluation
    """
    
    # INITIALISATION STRICTE avec metrics par défaut
    default_metric_value = 0.0
    default_metrics = {
        'accuracy': default_metric_value if task_type == 'classification' else None,
        'precision': default_metric_value if task_type == 'classification' else None,
        'recall': default_metric_value if task_type == 'classification' else None,
        'f1': default_metric_value if task_type == 'classification' else None,
        'r2': default_metric_value if task_type == 'regression' else None,
        'mae': default_metric_value if task_type == 'regression' else None,
        'mse': default_metric_value if task_type == 'regression' else None,
        'rmse': default_metric_value if task_type == 'regression' else None
    }
    
    result = {
        "model_name": model_name,
        "success": False,
        "model": None,
        "training_time": 0,
        "error": None,
        "best_params": None,
        "cv_scores": None,
        "metrics": default_metrics.copy()  
    }
    
    start_time = time.time()
    
    try:
        if monitor:
            monitor.start_model(model_name)
        
        cv_folds = TRAINING_CONSTANTS.get("CV_FOLDS", 5)
        random_state = TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
        n_jobs = TRAINING_CONSTANTS.get("N_JOBS", -1)
        
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = 'r2'
        
        # ========================================================================
        # PHASE 1: ENTRAÎNEMENT
        # ========================================================================
        model_trained = False
        
        if param_grid and len(param_grid) > 0:
            logger.info(f"🔍 Optimisation hyperparamètres pour {model_name}")
            
            max_combinations = TRAINING_CONSTANTS.get("MAX_GRID_COMBINATIONS", 100)
            total_combinations = np.prod([len(v) for v in param_grid.values()])
            
            if total_combinations > max_combinations:
                logger.warning(f"Grille réduite: {total_combinations} → limitation")
                limited_param_grid = {}
                for k, v in param_grid.items():
                    limited_param_grid[k] = v[:2] if len(v) > 2 else v
                param_grid = limited_param_grid
            
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring=scoring,
                n_jobs=n_jobs, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            result["model"] = grid_search.best_estimator_
            result["best_params"] = grid_search.best_params_
            result["cv_scores"] = {
                'mean': grid_search.best_score_,
                'std': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
            }
            result["success"] = True
            model_trained = True
            
            logger.info(f"✅ Optimisation OK {model_name} - CV score: {grid_search.best_score_:.3f}")
            
        else:
            try:
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
                result["cv_scores"] = {'mean': cv_scores.mean(), 'std': cv_scores.std()}
                logger.info(f"✅ CV {model_name}: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            except Exception as cv_error:
                logger.warning(f"⚠️ CV échouée {model_name}: {cv_error}")
                result["cv_scores"] = None
            
            pipeline.fit(X_train, y_train)
            result["model"] = pipeline
            result["success"] = True
            model_trained = True
            
            logger.info(f"✅ Entraînement OK {model_name}")
        
        result["training_time"] = time.time() - start_time
        
        # ========================================================================
        # PHASE 2: ÉVALUATION IMMÉDIATE avec fallback robuste
        # ========================================================================
        if model_trained and result["model"] is not None:
            try:
                logger.info(f"📊 Calcul métriques {model_name}")
                
                from src.evaluation.metrics import evaluate_single_train_test_split
                
                # AJOUT: Passer y_proba si disponible
                y_proba_test = None
                if hasattr(result["model"], 'predict_proba'):
                    try:
                        y_proba_test = result["model"].predict_proba(X_test)
                        logger.info(f"✅ {model_name}: Probabilités calculées ({y_proba_test.shape})")
                    except Exception as proba_error:
                        logger.warning(f"⚠️ {model_name}: predict_proba échoué: {proba_error}")
                
                evaluation_result = evaluate_single_train_test_split(
                    model=result["model"],
                    X_test=X_test,
                    y_test=y_test,
                    task_type=task_type,
                    label_encoder=None,
                    sample_metrics=True,
                    max_samples_metrics=100000
                )
                
                # FUSION INTELLIGENTE avec validation type
                if evaluation_result and isinstance(evaluation_result, dict):
                    if evaluation_result.get('success', False):
                        # EXTRACTION SÉCURISÉE des métriques (incluant None)
                        computed_metrics = {}
                        
                        for k, v in evaluation_result.items():
                            # Ignorer les métadonnées
                            if k in ['success', 'warnings', 'error', 'task_type', 'n_samples']:
                                continue
                            
                            # GARDER TOUTES les métriques, même None (pour debug)
                            if isinstance(v, (int, float, np.number)):
                                if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                                    computed_metrics[k] = float(v)
                                else:
                                    logger.warning(f"⚠️ {model_name}: {k} est NaN/Inf, ignoré")
                            elif v is None:
                                # GARDER None pour traçabilité
                                computed_metrics[k] = 0.0  # Fallback à 0.0
                                logger.warning(f"⚠️ {model_name}: {k} est None, fallback à 0.0")
                        
                        if computed_metrics:
                            result['metrics'] = computed_metrics
                            
                            # LOG DÉTAILLÉ pour debug
                            logger.info(f"✅ Métriques {model_name}: {list(computed_metrics.keys())}")
                            
                            # VÉRIFICATION CRITIQUE f1_score et roc_auc
                            if task_type == 'classification':
                                if 'f1_score' in computed_metrics:
                                    logger.info(f"   ✅ f1_score: {computed_metrics['f1_score']:.4f}")
                                else:
                                    logger.error(f"   ❌ f1_score MANQUANT!")
                                
                                if 'roc_auc' in computed_metrics:
                                    logger.info(f"   ✅ roc_auc: {computed_metrics['roc_auc']:.4f}")
                                else:
                                    logger.warning(f"   ⚠️ roc_auc MANQUANT (probablement pas de predict_proba)")
                        else:
                            logger.warning(f"⚠️ Aucune métrique valide calculée pour {model_name}")
                    else:
                        error_msg = evaluation_result.get('error', 'Évaluation échouée')
                        result['metrics']['error'] = error_msg
                        logger.warning(f"⚠️ Évaluation échouée {model_name}: {error_msg}")
                else:
                    logger.error(f"❌ Résultat évaluation invalide {model_name}: {type(evaluation_result)}")
                    result['metrics']['error'] = 'Résultat évaluation invalide'
                
            except Exception as eval_error:
                logger.error(f"❌ Erreur évaluation {model_name}: {eval_error}", exc_info=True)
                result['metrics'] = default_metrics.copy()
                result['metrics']['error'] = f'Erreur évaluation: {str(eval_error)[:100]}'
                
        # ========================================================================
        # VALIDATION FINALE STRICTE
        # ========================================================================
        if not result.get('metrics') or not isinstance(result['metrics'], dict):
            logger.error(f"❌ CRITIQUE {model_name}: metrics None ou invalide, réinitialisation")
            result['metrics'] = default_metrics.copy()
            result['metrics']['error'] = 'Metrics non calculées'
        
        # Vérification présence au moins UNE métrique valide
        valid_metrics_count = sum(
            1 for k, v in result['metrics'].items()
            if k != 'error' and v is not None and isinstance(v, (int, float, np.number))
        )
        
        if valid_metrics_count == 0:
            logger.warning(f"⚠️ {model_name}: Aucune métrique valide, ajout d'un indicateur")
            result['metrics']['valid_metrics_count'] = 0
        
        if monitor:
            resource_info = monitor.check_resources()
            logger.info(f"✅ {model_name} terminé en {result['training_time']:.2f}s, {valid_metrics_count} métriques valides")
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["training_time"] = time.time() - start_time
        
        # GARANTIE: metrics toujours présentes même en cas d'erreur
        if not result.get('metrics') or not isinstance(result['metrics'], dict):
            result["metrics"] = default_metrics.copy()
        
        result["metrics"]['error'] = str(e)[:200]
        result["metrics"]['training_failed'] = True
        
        logger.error(f"❌ Erreur critique {model_name}: {e}", exc_info=True)
    
    # ASSERTION FINALE (debug)
    assert 'metrics' in result, f"CRITIQUE: 'metrics' manquante dans result pour {model_name}"
    assert isinstance(result['metrics'], dict), f"CRITIQUE: metrics n'est pas un dict pour {model_name}"
    
    logger.debug(f"🔍 {model_name} - Clés result: {list(result.keys())}")
    logger.debug(f"🔍 {model_name} - Clés metrics: {list(result['metrics'].keys())}")
    
    return result


def train_single_model_unsupervised(
    model_name: str,
    pipeline,
    X,
    param_grid: Dict = None,
    monitor: Any = None
) -> Dict[str, Any]:
    """
    VERSION CORRIGÉE avec garantie labels à TOUS les niveaux
    """
    result = {
        "model_name": model_name,
        "success": False,
        "model": None,
        "training_time": 0.0,
        "error": None,
        "best_params": None,
        "labels": None,  
        "metrics": {}
    }

    start_time = time.time()

    try:
        X_arr, is_df, idx = _ensure_array_like(X)
        n_samples = X_arr.shape[0]
        if n_samples < 2:
            raise ValueError("Jeu de données insuffisant pour le clustering")

        # Identification de l'estimateur
        estimator = pipeline
        if hasattr(pipeline, "named_steps"):
            last_step = list(pipeline.named_steps.items())[-1]
            estimator = last_step[1]

        def _fit_predict_robust(pipeline_obj, X_in):
            """
            Fonction ULTRA-ROBUSTE pour obtenir les labels
            Essaie TOUTES les méthodes possibles
            """
            labels = None
            method_used = None
            
            try:
                # MÉTHODE 1: fit_predict (préféré pour clustering)
                if hasattr(pipeline_obj, "fit_predict"):
                    labels = pipeline_obj.fit_predict(X_in)
                    method_used = "fit_predict"
                    logger.debug(f"   Labels via fit_predict: {len(labels)}")
                    return labels, method_used
            except Exception as e:
                logger.warning(f"fit_predict échoué: {e}")
            
            try:
                # MÉTHODE 2: fit + labels_
                fitted = pipeline_obj.fit(X_in)
                if hasattr(fitted, "labels_"):
                    labels = getattr(fitted, "labels_")
                    method_used = "labels_"
                    logger.debug(f"   Labels via labels_: {len(labels)}")
                    return labels, method_used
            except Exception as e:
                logger.warning(f"fit + labels_ échoué: {e}")
            
            try:
                # MÉTHODE 3: fit + predict
                if not hasattr(pipeline_obj, "fit"):
                    raise AttributeError("Pas de méthode fit")
                
                fitted = pipeline_obj.fit(X_in)
                if hasattr(fitted, "predict"):
                    labels = fitted.predict(X_in)
                    method_used = "fit + predict"
                    logger.debug(f"   Labels via fit+predict: {len(labels)}")
                    return labels, method_used
            except Exception as e:
                logger.warning(f"fit + predict échoué: {e}")
            
            # MÉTHODE 4: Accès direct au dernier step du pipeline
            try:
                if hasattr(pipeline_obj, "named_steps"):
                    last_step_name = list(pipeline_obj.named_steps.keys())[-1]
                    last_estimator = pipeline_obj.named_steps[last_step_name]
                    
                    if hasattr(last_estimator, "labels_"):
                        labels = getattr(last_estimator, "labels_")
                        method_used = f"pipeline.{last_step_name}.labels_"
                        logger.debug(f"   Labels via {method_used}: {len(labels)}")
                        return labels, method_used
            except Exception as e:
                logger.warning(f"Accès dernier step échoué: {e}")
            
            raise RuntimeError("AUCUNE méthode pour obtenir les labels n'a fonctionné!")

        # ENTRAÎNEMENT avec/sans optimisation
        if param_grid and isinstance(param_grid, dict) and len(param_grid) > 0:
            from itertools import product
            from sklearn.base import clone
            
            keys = list(param_grid.keys())
            values = [param_grid[k] if isinstance(param_grid[k], (list, tuple, np.ndarray)) else [param_grid[k]] for k in keys]
            combos = list(product(*values))

            best_score = -np.inf
            best_params = None
            best_candidate = None
            best_labels = None
            best_metrics = None

            for i, combo in enumerate(combos):
                try:
                    params = dict(zip(keys, combo))
                    pipeline_candidate = clone(pipeline)
                    
                    try:
                        pipeline_candidate.set_params(**params)
                    except Exception as e:
                        if hasattr(pipeline_candidate, "named_steps"):
                            final_name = list(pipeline_candidate.named_steps.keys())[-1]
                            pipeline_candidate.named_steps[final_name].set_params(**params)
                        else:
                            raise e

                    # RÉCUPÉRATION LABELS ROBUSTE
                    labels, method = _fit_predict_robust(pipeline_candidate, X)
                    labels = np.asarray(labels)
                    
                    logger.debug(f"Combo {i+1}/{len(combos)}: labels obtenus via {method}")
                    
                    metrics = _safe_cluster_metrics(X, labels)
                    score = metrics.get("silhouette", np.nan)
                    
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_params = params
                        best_candidate = pipeline_candidate
                        best_labels = labels
                        best_metrics = metrics
                        
                except Exception as e:
                    logger.warning(f"⚠️ Combinaison {i+1}/{len(combos)} échouée: {e}")
                    continue

            if best_score == -np.inf:
                raise RuntimeError("Aucun jeu de paramètres valides trouvé")
                
            result["best_params"] = best_params
            result["model"] = best_candidate
            result["labels"] = best_labels
            result["metrics"] = best_metrics

        else:
            # Entraînement sans optimisation
            labels, method = _fit_predict_robust(pipeline, X)
            labels = np.asarray(labels)
            
            logger.info(f"Labels obtenus via {method}")
            
            result["model"] = pipeline
            result["labels"] = labels
            result["metrics"] = _safe_cluster_metrics(X_arr, labels)

        result["training_time"] = time.time() - start_time
        result["success"] = True

        # VALIDATION FINALE STRICTE
        if result["labels"] is None:
            raise RuntimeError("CRITIQUE: Labels sont None après entraînement!")
        
        if not isinstance(result["labels"], (np.ndarray, list)):
            raise TypeError(f"CRITIQUE: Labels type invalide: {type(result['labels'])}")
        
        if len(result["labels"]) != n_samples:
            raise ValueError(
                f"CRITIQUE: Incohérence nb labels ({len(result['labels'])}) "
                f"vs nb samples ({n_samples})"
            )
        
        logger.info(
            f"✅ {model_name}: Clustering réussi\n"
            f"   • Labels: {len(result['labels'])} points\n"
            f"   • Clusters uniques: {len(np.unique(result['labels']))}\n"
            f"   • Silhouette: {result['metrics'].get('silhouette', 'N/A')}"
        )

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["training_time"] = time.time() - start_time
        logger.error(f"❌ Erreur clustering {model_name}: {e}", exc_info=True)

    return result

# ===============================================
# FONCTION PRINCIPALE AMÉLIORÉE - CORRIGÉE
# ===============================================

def log_structured(level: str, message: str, extra: Dict = None):
    """Fonction de journalisation structurée avec format texte clair."""
    try:
        # Format de base du message
        log_message = f"{message}"
        
        # Ajouter les métadonnées extra de manière lisible
        if extra:
            extra_str = " ".join([f"[{key}: {value}]" for key, value in extra.items()])
            log_message = f"{log_message} {extra_str}"
        
        # Journaliser avec le niveau approprié
        logger.log(getattr(logging, level.upper()), log_message)
    except Exception as e:
        logger.error(f"Erreur lors de la journalisation structurée: {str(e)[:100]}")

# =================================
# FONCTION DE STOCKAGE POUR SESSION
# =================================
def _store_results_in_session(results: List[Dict], mlflow_runs: List[Dict]) -> bool:
    """
    Stockage ATOMIQUE avec déduplication et validation stricte.
    🆕 Synchronisation STATE + session_state garantie
    """
    try:
        logger.info(f"🔄 Démarrage stockage session: {len(results)} résultats, {len(mlflow_runs)} runs MLflow")
        
        if not STREAMLIT_AVAILABLE or st is None:
            logger.warning("⚠️ Streamlit non disponible - stockage limité à STATE")
            # Fallback STATE uniquement
            if hasattr(STATE, 'mlflow_runs') and mlflow_runs:
                existing_ids = {r.get('run_id') for r in STATE.mlflow_runs if r.get('run_id')}
                new_runs = [r for r in mlflow_runs if r.get('run_id') not in existing_ids]
                STATE.mlflow_runs.extend(new_runs)
                logger.info(f"✅ {len(new_runs)} runs ajoutés à STATE (fallback)")
            return False
        
        # 🔒 VALIDATION STRICTE des données
        valid_results = []
        for r in results:
            if not isinstance(r, dict):
                logger.warning(f"⚠️ Résultat ignoré (type invalide): {type(r)}")
                continue
            if not r.get('model_name'):
                logger.warning(f"⚠️ Résultat ignoré (model_name manquant): {list(r.keys())}")
                continue
            valid_results.append(r)
        
        valid_mlflow_runs = []
        for r in mlflow_runs:
            if not isinstance(r, dict):
                logger.warning(f"⚠️ Run MLflow ignoré (type invalide): {type(r)}")
                continue
            if not r.get('run_id'):
                logger.warning(f"⚠️ Run MLflow ignoré (run_id manquant)")
                continue
            valid_mlflow_runs.append(r)
        
        logger.info(f"✅ Validation: {len(valid_results)}/{len(results)} résultats, {len(valid_mlflow_runs)}/{len(mlflow_runs)} runs valides")
        
        # 🔧 INITIALISATION ATOMIQUE
        if 'ml_results' not in st.session_state:
            st.session_state.ml_results = []
        if 'mlflow_runs' not in st.session_state:
            st.session_state.mlflow_runs = []
        
        # 🎯 DÉDUPLICATION par run_id
        existing_run_ids = {r.get('run_id') for r in st.session_state.mlflow_runs if r.get('run_id')}
        new_runs = [r for r in valid_mlflow_runs if r.get('run_id') not in existing_run_ids]
        
        if len(new_runs) < len(valid_mlflow_runs):
            logger.info(f"ℹ️ {len(valid_mlflow_runs) - len(new_runs)} runs déjà existants (ignorés)")
        
        # 💾 STOCKAGE MULTI-NIVEAU avec transaction atomique
        try:
            # Niveau 1: Session Streamlit (prioritaire)
            st.session_state.ml_results.extend(valid_results)
            st.session_state.mlflow_runs.extend(new_runs)
            
            # Niveau 2: STATE global (synchronisation)
            if hasattr(STATE, 'ml_results'):
                if not isinstance(STATE.ml_results, list):
                    STATE.ml_results = []
                STATE.ml_results.extend(valid_results)
            
            if hasattr(STATE, 'mlflow_runs'):
                if not isinstance(STATE.mlflow_runs, list):
                    STATE.mlflow_runs = []
                
                state_existing_ids = {r.get('run_id') for r in STATE.mlflow_runs if r.get('run_id')}
                state_new_runs = [r for r in new_runs if r.get('run_id') not in state_existing_ids]
                STATE.mlflow_runs.extend(state_new_runs)
            
            # Niveau 3: TrainingState (si disponible)
            if hasattr(STATE, 'training') and hasattr(STATE.training, 'mlflow_runs'):
                if not isinstance(STATE.training.mlflow_runs, list):
                    STATE.training.mlflow_runs = []
                
                training_existing_ids = {r.get('run_id') for r in STATE.training.mlflow_runs if r.get('run_id')}
                training_new_runs = [r for r in new_runs if r.get('run_id') not in training_existing_ids]
                STATE.training.mlflow_runs.extend(training_new_runs)
            
            # ✅ VALIDATION POST-STOCKAGE
            total_session = len(st.session_state.mlflow_runs)
            total_state = len(STATE.mlflow_runs) if hasattr(STATE, 'mlflow_runs') else 0
            total_training = len(STATE.training.mlflow_runs) if hasattr(STATE, 'training') and hasattr(STATE.training, 'mlflow_runs') else 0
            
            logger.info(
                f"✅ Stockage réussi:\n"
                f"   • Résultats ajoutés: {len(valid_results)}\n"
                f"   • Nouveaux runs MLflow: {len(new_runs)}\n"
                f"   • Total session: {total_session} runs\n"
                f"   • Total STATE: {total_state} runs\n"
                f"   • Total training: {total_training} runs"
            )
            
            return True
            
        except Exception as storage_error:
            logger.error(f"❌ Erreur transaction stockage: {storage_error}", exc_info=True)
            # Rollback tentative (best effort)
            try:
                st.session_state.ml_results = st.session_state.ml_results[:-len(valid_results)]
                st.session_state.mlflow_runs = st.session_state.mlflow_runs[:-len(new_runs)]
            except:
                pass
            return False
        
    except Exception as e:
        logger.error(f"❌ Échec critique stockage session: {e}", exc_info=True)
        return False


# ===============================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT AVEC MLFLOW
# ===============================================
def train_single_model_with_mlflow(
    model_name: str,
    task_type: str,
    X_train: Optional[pd.DataFrame],
    y_train: Optional[pd.Series],
    X_test: Optional[pd.DataFrame],
    y_test: Optional[pd.Series],
    X: Optional[pd.DataFrame],
    column_types: Dict,
    preprocessing_choices: Dict,
    use_smote: bool,
    optimize: bool,
    feature_list: List[str],
    git_info: Dict,
    label_encoder: Any,
    sample_metrics: bool,
    max_samples_metrics: int,
    monitor: TrainingMonitor,
    mlflow_enabled: bool
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    VERSION CORRIGÉE avec :
    - Validation SMOTE stricte AVANT création pipeline
    - Synchronisation MLflow garantie
    - Métriques obligatoires
    """
    
    mlflow_run_data = None
    result = None
    
    try:
        # INJECTION feature_list
        preprocessing_choices['feature_list'] = feature_list
        logger.debug(f"✅ {model_name}: feature_list injectée ({len(feature_list)} features)")
        
        # VALIDATION données
        if task_type != 'clustering' and X_train is not None:
            missing_features = [f for f in feature_list if f not in X_train.columns]
            if missing_features:
                logger.error(f"❌ {model_name}: Features manquantes: {missing_features[:5]}")
                return None, None
            
            X_train = X_train[feature_list].copy()
            X_test = X_test[feature_list].copy()
            logger.debug(f"✅ {model_name}: X_train/X_test filtrés ({X_train.shape})")
        
        elif task_type == 'clustering' and X is not None:
            missing_features = [f for f in feature_list if f not in X.columns]
            if missing_features:
                logger.error(f"❌ {model_name}: Features manquantes dans X: {missing_features[:5]}")
                return None, None
            
            X = X[feature_list].copy()
            logger.debug(f"✅ {model_name}: X clustering filtré ({X.shape})")
        
        # FILTRAGE column_types
        feature_set = set(feature_list)
        filtered_column_types = {
            col_type: [col for col in cols if col in feature_set]
            for col_type, cols in column_types.items()
            if any(col in feature_set for col in cols)
        }
        
        logger.info(f"✅ {model_name}: Column types filtrés: {[(k, len(v)) for k, v in filtered_column_types.items()]}")
        
        # 🆕 VALIDATION SMOTE STRICTE (CRITIQUE)
        final_use_smote = False
        smote_validation_passed = True
        
        if use_smote and task_type == 'classification' and y_train is not None:
            # Récupération paramètres SMOTE
            smote_k = preprocessing_choices.get('smote_k_neighbors', 5)
            
            # VALIDATION classe minoritaire
            class_counts = y_train.value_counts()
            min_class_count = class_counts.min()
            
            if min_class_count <= smote_k:
                logger.error(
                    f"❌ {model_name}: SMOTE IMPOSSIBLE!\n"
                    f"   k_neighbors={smote_k} >= min_class_count={min_class_count}\n"
                    f"   SMOTE nécessite min_class_count > k_neighbors"
                )
                smote_validation_passed = False
                use_smote = False
                
                # Injection warning dans preprocessing_choices
                if 'smote_validation_error' not in preprocessing_choices:
                    preprocessing_choices['smote_validation_error'] = (
                        f"Classe minoritaire trop petite ({min_class_count} ≤ k={smote_k})"
                    )
            else:
                logger.info(
                    f"✅ {model_name}: SMOTE validé (k={smote_k} < min_class={min_class_count})"
                )
                final_use_smote = True
        
        # Mise à jour preprocessing_choices avec validation
        preprocessing_choices['use_smote'] = final_use_smote
        preprocessing_choices['smote_validation_passed'] = smote_validation_passed
        
        # CRÉATION pipeline (avec SMOTE validé)
        pipeline, param_grid = create_leak_free_pipeline(
            model_name=model_name,
            task_type=task_type,
            column_types=filtered_column_types,
            preprocessing_choices=preprocessing_choices,
            use_smote=final_use_smote,  # 🎯 SMOTE validé
            optimize_hyperparams=optimize
        )
        
        if pipeline is None:
            logger.error(f"❌ {model_name}: Pipeline vide")
            return None, None

        # RÉCUPÉRATION IMBALANCE INFO depuis STATE
        imbalance_ratio = None
        imbalance_level = None
        min_class_count = None
        
        if hasattr(STATE, 'imbalance_config') and isinstance(STATE.imbalance_config, dict):
            imbalance_ratio = STATE.imbalance_config.get('imbalance_ratio')
            imbalance_level = STATE.imbalance_config.get('imbalance_level')
            min_class_count = STATE.imbalance_config.get('min_class_count')
        
        # CONFIGURATION MLFLOW ENRICHIE
        run_id = None
        timestamp = int(time.time())
        
        if mlflow_enabled:
            try:
                run_name = f"{clean_model_name(model_name)}_{timestamp}"
                mlflow.start_run(run_name=run_name)
                
                # PARAMS STANDARDS
                mlflow.log_param("task_type", task_type)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("optimize_hyperparams", optimize)
                mlflow.log_param("n_features", len(feature_list))
                
                # 🆕 PARAMS SMOTE AVEC VALIDATION
                mlflow.log_param("use_smote", final_use_smote)
                mlflow.log_param("smote_validation_passed", smote_validation_passed)
                
                if final_use_smote:
                    mlflow.log_param("smote_k_neighbors", preprocessing_choices.get('smote_k_neighbors', 5))
                    mlflow.set_tag("smote_applied", "true")
                else:
                    mlflow.set_tag("smote_applied", "false")
                    if not smote_validation_passed:
                        mlflow.set_tag("smote_validation_error", 
                                     preprocessing_choices.get('smote_validation_error', 'Unknown'))
                
                if imbalance_ratio is not None:
                    mlflow.log_metric("imbalance_ratio", float(imbalance_ratio))
                    mlflow.set_tag("imbalance_level", str(imbalance_level))
                
                if min_class_count is not None:
                    mlflow.log_metric("min_class_count", int(min_class_count))
                
                # PARAMS GIT
                for k, v in git_info.items():
                    if v:
                        mlflow.log_param(f"git_{k}", v)
                
                # PARAMS PREPROCESSING
                for k, v in preprocessing_choices.items():
                    if isinstance(v, (str, int, float, bool)):
                        mlflow.log_param(f"preprocessing_{k}", v)
                
                # TAGS ADDITIONNELS
                mlflow.set_tag("framework", "scikit-learn")
                mlflow.set_tag("app_version", "3.0.0")
                
                run_id = mlflow.active_run().info.run_id
                logger.info(f"✅ {model_name}: Run MLflow démarré ({run_id})")
                
            except Exception as e:
                logger.warning(f"⚠️ {model_name}: Échec démarrage MLflow: {e}")
                mlflow_enabled = False

        # ENTRAÎNEMENT
        training_result = None
        try:
            if task_type == 'clustering':
                training_result = train_single_model_unsupervised(
                    model_name=model_name,
                    pipeline=pipeline,
                    X=X,
                    param_grid=param_grid,
                    monitor=monitor
                )
            else:
                training_result = train_single_model_supervised(
                    model_name=model_name,
                    pipeline=pipeline,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    param_grid=param_grid,
                    task_type=task_type,
                    monitor=monitor
                )
        except Exception as e:
            logger.error(f"❌ {model_name}: Échec entraînement: {e}")
            if mlflow_enabled and mlflow.active_run():
                mlflow.log_param("training_error", str(e)[:200])
                mlflow.end_run(status="FAILED")
            return None, None

        # 🆕 VALIDATION STRICTE DES MÉTRIQUES (CRITIQUE)
        if not training_result or not training_result.get('success'):
            logger.error(f"❌ {model_name}: Entraînement échoué")
            if mlflow_enabled and mlflow.active_run():
                mlflow.end_run(status="FAILED")
            return None, None
        
        if 'metrics' not in training_result or not isinstance(training_result['metrics'], dict):
            logger.error(f"❌ {model_name}: CRITIQUE - metrics manquantes ou invalides")
            training_result['metrics'] = {'error': 'Metrics non générées'}
            if mlflow_enabled and mlflow.active_run():
                mlflow.end_run(status="FAILED")
            return None, None

        # ÉVALUATION (déjà faite dans train_single_model_supervised/unsupervised)
        metrics = training_result['metrics']
        warnings_list = metrics.pop("warnings", [])

        # SAUVEGARDE modèle
        model_name_clean = clean_model_name(model_name)
        model_filename = f"{model_name_clean}_{task_type}_{timestamp}.joblib"
        model_path = os.path.join("models_output", model_filename)
        
        try:
            os.makedirs("models_output", exist_ok=True)
            joblib.dump(training_result["model"], model_path)
            logger.info(f"✅ {model_name}: Modèle sauvegardé ({model_path})")
        except Exception as e:
            logger.error(f"❌ {model_name}: Échec sauvegarde: {e}")

        # LOGGING MLFLOW ENRICHI
        # LOGGING MLFLOW ENRICHI
        if mlflow_enabled:
            try:
                # MÉTRIQUES PRINCIPALES
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        mlflow.log_metric(k, float(v))
                        
                mlflow.log_metric("training_time", training_result.get("training_time", 0.0))
                        
                # MÉTRIQUES SMOTE
                if final_use_smote:
                    mlflow.log_metric("smote_applied", 1.0)
                    mlflow.log_metric("data_augmentation", 1.0)
                else:
                    mlflow.log_metric("smote_applied", 0.0)
                        
                if not smote_validation_passed:
                    mlflow.log_metric("smote_validation_failed", 1.0)
                        
                if imbalance_ratio is not None and imbalance_ratio > 2:
                    mlflow.log_metric("imbalance_severity", min(10.0, imbalance_ratio / 2))
                        
                # ARTEFACTS
                mlflow.log_artifact(model_path)
                        
                # TAGS FINAUX
                if metrics.get('accuracy'):
                    mlflow.set_tag("accuracy_tier", 
                                "excellent" if metrics['accuracy'] > 0.9 else 
                                "good" if metrics['accuracy'] > 0.8 else 
                                "fair" if metrics['accuracy'] > 0.7 else "poor")
                        
                # 🎯 FORMATAGE ET COLLECTE IMMÉDIATE (AVANT end_run)
                mlflow_run_data = format_mlflow_run_for_ui(
                    run_info=mlflow.active_run(),
                    metrics=metrics,
                    preprocessing_choices=preprocessing_choices,
                    model_name=model_name,
                    timestamp=timestamp
                )
                        
                # ENRICHISSEMENT
                if mlflow_run_data:
                    mlflow_run_data['smote_applied'] = final_use_smote
                    mlflow_run_data['smote_validation_passed'] = smote_validation_passed
                    mlflow_run_data['imbalance_ratio'] = imbalance_ratio
                    mlflow_run_data['imbalance_level'] = imbalance_level
                            
                    # 🎯 COLLECTE ATOMIQUE dans le collecteur global
                    collected = MLFLOW_COLLECTOR.add_run(mlflow_run_data)
                    if collected:
                        logger.info(f"✅ {model_name}: Run collecté dans MLFLOW_COLLECTOR")
                    else:
                        logger.warning(f"⚠️ {model_name}: Run déjà collecté ou invalide")
                        
                logger.info(f"✅ {model_name}: Run MLflow complété")
                        
            except Exception as e:
                logger.warning(f"⚠️ {model_name}: Échec logging MLflow: {e}")
            finally:
                if mlflow.active_run():
                    mlflow.end_run()

        # 🆕 CONSTRUCTION RÉSULTAT FINAL ROBUSTE
        result = {
            "model_name": model_name,
            "task_type": task_type,
            "metrics": metrics,  # Déjà validées
            "training_time": training_result.get("training_time", 0),
            "model_path": model_path,
            "warnings": warnings_list,
            "success": True,  # On arrive ici = succès
            "feature_names": feature_list,
            "smote_applied": final_use_smote,
            "smote_validation_passed": smote_validation_passed,
            "imbalance_ratio": imbalance_ratio,
            "n_features": len(feature_list)
        }

        # Ajout données spécifiques au task
        if task_type == 'clustering':
            if training_result.get("labels") is not None:
                result["labels"] = training_result["labels"]
                result["X_sample"] = X
                logger.info(f"✅ {model_name}: Labels clustering sauvegardés ({len(result['labels'])} points)")
            else:
                logger.error(f"❌ {model_name}: CRITIQUE - Labels clustering manquants!")
                result["warnings"].append("Labels clustering non disponibles")
        
        else:  # Supervisé
            result["X_train"] = X_train
            result["y_train"] = y_train
            result["X_test"] = X_test
            result["y_test"] = y_test
            result["model"] = training_result["model"]
            
            if X_test is not None and len(X_test) > 0:
                sample_size = min(1000, len(X_test))
                result["X_sample"] = X_test.iloc[:sample_size]

    except Exception as e:
        logger.error(f"❌ {model_name}: Erreur critique: {e}", exc_info=True)
        if mlflow_enabled and mlflow.active_run():
            try:
                mlflow.log_param("critical_error", str(e)[:200])
            except:
                pass
            mlflow.end_run(status="FAILED")
        return None, None

    return result, mlflow_run_data


# ======================================
# 🔧 TRAIN_MODELS - FONCTION PRINCIPALE
# ======================================

def train_models(
    df: pd.DataFrame,
    target_column: Optional[str],
    model_names: List[str],
    task_type: str,
    test_size: float = 0.2,
    optimize: bool = False,
    feature_list: List[str] = None,
    use_smote: bool = False,
    preprocessing_choices: Dict = None,
    sample_metrics: bool = True,
    max_samples_metrics: int = 100000,
    n_jobs: int = None
) -> List[Dict[str, Any]]:
    """
    🆕 VERSION CORRIGÉE avec:
    - Validation SMOTE précoce
    - Synchronisation MLflow GARANTIE multi-niveaux
    - Métriques obligatoires
    - Logs structurés
    """
    
    if n_jobs is None:
        n_jobs = TRAINING_CONSTANTS.get("N_JOBS", -1)
    
    with TRAINING_STATE.training_session() as state:
        results = []
        monitor = TrainingMonitor()
        monitor.start_training()

        # ========================================================================
        # PHASE 1: VALIDATION FEATURE_LIST
        # ========================================================================
        if not feature_list or len(feature_list) == 0:
            logger.error("❌ feature_list vide!")
            
            if target_column and target_column in df.columns:
                feature_list = [col for col in df.columns if col != target_column]
            else:
                feature_list = list(df.columns)
            
            if not feature_list:
                error_msg = "Impossible de déterminer feature_list"
                logger.error(f"❌ {error_msg}")
                return [{"model_name": "Validation", "metrics": {"error": error_msg}, "warnings": [], "success": False}]
            
            logger.warning(f"⚠️ feature_list récupérée auto: {len(feature_list)} features")
        
        logger.info(f"✅ feature_list validée: {len(feature_list)} features")

        # ========================================================================
        # PHASE 2: VALIDATION TASK TYPE
        # ========================================================================
        task_type = task_type.lower()
        if task_type not in ['classification', 'regression', 'clustering', 'unsupervised']:
            logger.error(f"❌ Type invalide: {task_type}")
            return [{"model_name": "Validation", "metrics": {"error": f"Type {task_type} non supporté"}, "warnings": [], "success": False}]
        
        if task_type == 'unsupervised':
            task_type = 'clustering'

        if task_type == 'clustering':
            target_column = None
            use_smote = False
            test_size = 0.0

        logger.info(f"🚀 Début entraînement: {task_type}, {len(model_names)} modèles, {len(feature_list)} features")

        # ========================================================================
        # PHASE 3: VALIDATION DONNÉES
        # ========================================================================
        min_samples = TRAINING_CONSTANTS.get("MIN_SAMPLES_REQUIRED", 10)
        if len(df) < min_samples:
            logger.error(f"❌ Échantillons insuffisants: {len(df)} < {min_samples}")
            return [{"model_name": "Validation", "metrics": {"error": "Échantillons insuffisants"}, "warnings": [], "success": False}]

        # ========================================================================
        # PHASE 4: CONFIGURATION MLFLOW
        # ========================================================================
        mlflow_enabled = MLFLOW_AVAILABLE and MLFLOW_CONSTANTS.get("AVAILABLE", False)
        git_info = get_git_info() if mlflow_enabled else {}

        if mlflow_enabled:
            try:
                from utils.mlflow import configure_mlflow
                configured = configure_mlflow(
                    MLFLOW_CONSTANTS.get("TRACKING_URI", "sqlite:///mlflow.db"),
                    MLFLOW_CONSTANTS.get("EXPERIMENT_NAME", "datalab_experiments")
                )
                if not configured:
                    raise RuntimeError("Échec config MLflow")
                logger.info(f"✅ MLflow configuré: {MLFLOW_CONSTANTS.get('EXPERIMENT_NAME')}")
            except Exception as e:
                logger.warning(f"⚠️ Échec config MLflow: {e}")
                mlflow_enabled = False

        # ========================================================================
        # PHASE 5: PREPROCESSING CONFIG
        # ========================================================================
        if preprocessing_choices is None:
            preprocessing_choices = {
                'numeric_imputation': PREPROCESSING_CONSTANTS.get("NUMERIC_IMPUTATION_DEFAULT", "mean"),
                'categorical_imputation': PREPROCESSING_CONSTANTS.get("CATEGORICAL_IMPUTATION_DEFAULT", "most_frequent"),
                'remove_constant_cols': True,
                'remove_identifier_cols': True,
                'scale_features': True,
                'scaling_method': PREPROCESSING_CONSTANTS.get("SCALING_METHOD", "standard"),
                'encoding_method': PREPROCESSING_CONSTANTS.get("ENCODING_METHOD", "onehot"),
                'random_state': TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
            }

        os.makedirs("models_output", exist_ok=True)
        os.makedirs(LOGGING_CONSTANTS.get("LOG_DIR", "logs"), exist_ok=True)

        # ========================================================================
        # PHASE 6: PRÉPARATION DONNÉES
        # ========================================================================
        X = df[feature_list].copy()
        if X.empty:
            logger.error("❌ DataFrame vide après sélection features")
            return [{"model_name": "Validation", "metrics": {"error": "DataFrame vide"}, "warnings": [], "success": False}]

        y, label_encoder = None, None
        if task_type != 'clustering' and target_column:
            y_raw = df[target_column].copy()
            y_encoded, label_encoder, warnings_enc = safe_label_encode(y_raw)
            y = pd.Series(y_encoded, index=y_raw.index, name=target_column)
            if warnings_enc:
                logger.warning(f"⚠️ Encodage labels: {warnings_enc}")

        data_validation = DataValidator.validate_training_data(X, y, task_type)
        if not data_validation["is_valid"]:
            logger.error(f"❌ Validation données échouée: {', '.join(data_validation['issues'])}")
            return [{"model_name": "Validation", "metrics": {"error": ', '.join(data_validation['issues'])}, "warnings": data_validation.get('warnings', []), "success": False}]

        column_types = auto_detect_column_types(X)

        # ========================================================================
        # PHASE 7: 🆕 VALIDATION SMOTE PRÉCOCE (AVANT SPLIT)
        # ========================================================================
        smote_validation_global = {
            'requested': use_smote,
            'validated': False,
            'reason': None
        }
        
        if use_smote and task_type == 'classification' and y is not None:
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            smote_k = preprocessing_choices.get('smote_k_neighbors', 5)
            
            if min_class_count <= smote_k:
                logger.error(
                    f"❌ SMOTE GLOBAL IMPOSSIBLE!\n"
                    f"   Classe minoritaire: {min_class_count} échantillons\n"
                    f"   k_neighbors: {smote_k}\n"
                    f"   SMOTE nécessite min_class_count > k_neighbors\n"
                    f"   → SMOTE sera DÉSACTIVÉ pour TOUS les modèles"
                )
                use_smote = False
                smote_validation_global['validated'] = False
                smote_validation_global['reason'] = f"min_class_count ({min_class_count}) ≤ k ({smote_k})"
                
                # Warning global
                logger.warning("⚠️ DÉSACTIVATION SMOTE pour tous les modèles")
            else:
                smote_validation_global['validated'] = True
                smote_validation_global['reason'] = f"Validation OK (min_class={min_class_count} > k={smote_k})"
                logger.info(f"✅ SMOTE global validé: {smote_validation_global['reason']}")
        
        # Injection dans preprocessing_choices
        preprocessing_choices['use_smote'] = use_smote
        preprocessing_choices['smote_validation_global'] = smote_validation_global

        # ========================================================================
        # PHASE 8: SPLIT TRAIN/TEST
        # ========================================================================
        X_train = X_test = y_train = y_test = None
        if task_type != 'clustering':
            random_state = TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
            stratify = y if task_type == 'classification' else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=stratify
                )
                logger.info(f"✅ Split train/test: {len(X_train)} train, {len(X_test)} test")
            except Exception as e:
                logger.error(f"❌ Échec split: {e}")
                return [{"model_name": "Validation", "metrics": {"error": f"Split échoué: {e}"}, "warnings": [], "success": False}]

        # ========================================================================
        # PHASE 9: ENTRAÎNEMENT AVEC WRAPPER
        # ========================================================================
        successful_models = 0

        def train_model_wrapper(args):
            i, model_name = args
            logger.info(f"🔧 Début entraînement {model_name} ({i}/{len(model_names)})")
            
            result, mlflow_data = train_single_model_with_mlflow(
                model_name=model_name,
                task_type=task_type,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                X=X,
                column_types=column_types,
                preprocessing_choices=preprocessing_choices,
                use_smote=use_smote,
                optimize=optimize,
                feature_list=feature_list,
                git_info=git_info,
                label_encoder=label_encoder,
                sample_metrics=sample_metrics,
                max_samples_metrics=max_samples_metrics,
                monitor=monitor,
                mlflow_enabled=mlflow_enabled
            )
            
            # 🎯 STOCKAGE MLFLOW DANS STATE (CRITIQUE)
            if mlflow_data and isinstance(mlflow_data, dict):
                state.mlflow_collector.add_run(mlflow_data)
                logger.info(f"✅ MLflow run ajouté à collector: {mlflow_data.get('run_id', 'N/A')[:8]}")
            else:
                logger.warning(f"⚠️ Pas de mlflow_data pour {model_name}")
            
            return result

        try:
            model_args = [(i, model_name) for i, model_name in enumerate(model_names, 1)]
            
            if n_jobs == 1 or len(model_names) == 1:
                logger.info("🔄 Exécution séquentielle")
                parallel_results = [train_model_wrapper(args) for args in model_args]
            else:
                logger.info(f"🔄 Exécution parallèle (n_jobs={n_jobs})")
                parallel_results = Parallel(n_jobs=n_jobs)(
                    delayed(train_model_wrapper)(args) for args in model_args
                )
            
            results = [res for res in parallel_results if res is not None and res.get("success", False)]
            successful_models = len(results)
            
        except Exception as e:
            logger.error(f"❌ Échec exécution parallèle: {e}")
            results = []
            for i, model_name in enumerate(model_names, 1):
                try:
                    result = train_model_wrapper((i, model_name))
                    if result and result.get("success", False):
                        results.append(result)
                        successful_models += 1
                except Exception as model_error:
                    logger.error(f"❌ Échec {model_name}: {model_error}")

        # ========================================================================
        # PHASE 10: 🆕 SYNCHRONISATION MLFLOW MULTI-NIVEAUX GARANTIE
        # ========================================================================
        total_time = monitor.get_total_duration()
        mlflow_runs = state.mlflow_collector.get_runs()

        logger.info(f"✅ Entraînement terminé: {successful_models}/{len(model_names)} modèles, {total_time:.2f}s")
        logger.info(f"📊 MLflow runs collectés: {len(mlflow_runs)}")

        # 🎯 SYNCHRONISATION ATOMIQUE MULTI-NIVEAUX (CRITIQUE)
        if mlflow_runs:
            logger.info(f"🔄 Synchronisation MLflow de {len(mlflow_runs)} runs...")
            
            try:
                # VALIDATION des runs
                valid_mlflow_runs = []
                for run in mlflow_runs:
                    if isinstance(run, dict) and run.get('run_id'):
                        valid_mlflow_runs.append(run)
                    else:
                        logger.warning(f"⚠️ Run MLflow invalide ignoré: {type(run)}")
                
                if not valid_mlflow_runs:
                    logger.error("❌ Aucun run MLflow valide à synchroniser")
                else:
                    logger.info(f"✅ {len(valid_mlflow_runs)} runs MLflow valides")
                    
                    # === SYNCHRONISATION NIVEAU 1: session_state ===
                    sync_session = 0
                    try:
                        if STREAMLIT_AVAILABLE and st is not None:
                            if not hasattr(st.session_state, 'mlflow_runs'):
                                st.session_state.mlflow_runs = []
                            
                            existing_ids_session = {
                                r.get('run_id') 
                                for r in st.session_state.mlflow_runs 
                                if isinstance(r, dict) and r.get('run_id')
                            }
                            
                            new_runs_session = [
                                r for r in valid_mlflow_runs 
                                if r.get('run_id') not in existing_ids_session
                            ]
                            
                            if new_runs_session:
                                st.session_state.mlflow_runs.extend(new_runs_session)
                                sync_session = len(new_runs_session)
                                logger.info(
                                    f"✅ {sync_session} runs → session_state.mlflow_runs "
                                    f"(total: {len(st.session_state.mlflow_runs)})"
                                )
                    except Exception as e:
                        logger.error(f"❌ Erreur sync session_state: {e}", exc_info=True)
                    
                    # === SYNCHRONISATION NIVEAU 2: STATE.mlflow_runs ===
                    sync_state = 0
                    try:
                        from monitoring.state_managers import STATE
                        
                        existing_ids_state = {
                            r.get('run_id') 
                            for r in STATE.mlflow_runs 
                            if isinstance(r, dict) and r.get('run_id')
                        }
                        
                        new_runs_state = [
                            r for r in valid_mlflow_runs 
                            if r.get('run_id') not in existing_ids_state
                        ]
                        
                        if new_runs_state:
                            for run in new_runs_state:
                                STATE.add_mlflow_run(run)
                            sync_state = len(new_runs_state)
                            logger.info(
                                f"✅ {sync_state} runs → STATE.mlflow_runs "
                                f"(total: {len(STATE.mlflow_runs)})"
                            )
                    except Exception as e:
                        logger.error(f"❌ Erreur sync STATE: {e}", exc_info=True)
                    
                    # === SYNCHRONISATION NIVEAU 3: STATE.training ===
                    sync_training = 0
                    try:
                        if hasattr(st.session_state, 'training'):
                            if not hasattr(st.session_state.training, 'mlflow_runs'):
                                st.session_state.training.mlflow_runs = []
                            
                            existing_ids_training = {
                                r.get('run_id') 
                                for r in st.session_state.training.mlflow_runs 
                                if isinstance(r, dict) and r.get('run_id')
                            }
                            
                            new_runs_training = [
                                r for r in valid_mlflow_runs 
                                if r.get('run_id') not in existing_ids_training
                            ]
                            
                            if new_runs_training:
                                st.session_state.training.mlflow_runs.extend(new_runs_training)
                                sync_training = len(new_runs_training)
                                logger.info(
                                    f"✅ {sync_training} runs → STATE.training.mlflow_runs "
                                    f"(total: {len(st.session_state.training.mlflow_runs)})"
                                )
                    except Exception as e:
                        logger.error(f"❌ Erreur sync STATE.training: {e}", exc_info=True)
                    
                    # === RÉCAPITULATIF ===
                    total_synchronized = sync_session + sync_state + sync_training
                    
                    logger.info(
                        f"\n{'='*60}\n"
                        f"SYNCHRONISATION MLFLOW TERMINÉE\n"
                        f"{'='*60}\n"
                        f"Runs valides: {len(valid_mlflow_runs)}\n"
                        f"session_state: +{sync_session}\n"
                        f"STATE: +{sync_state}\n"
                        f"STATE.training: +{sync_training}\n"
                        f"Total synchronisé: {total_synchronized}\n"
                        f"{'='*60}"
                    )
                    
                    # === STOCKAGE LEGACY (pour compatibilité) ===
                    try:
                        storage_success = _store_results_in_session(results, valid_mlflow_runs)
                        if storage_success:
                            logger.info("✅ Stockage legacy session réussi")
                        else:
                            logger.warning("⚠️ Stockage legacy session échoué (non critique)")
                    except Exception as e:
                        logger.warning(f"⚠️ Stockage legacy: {e}")
            
            except Exception as sync_error:
                logger.error(f"❌ Erreur critique synchronisation MLflow: {sync_error}", exc_info=True)

        else:
            logger.warning("⚠️ Aucun run MLflow collecté")

        # ========================================================================
        # PHASE 11: VÉRIFICATION MLFLOW DB
        # ========================================================================
        if mlflow_enabled:
            try:
                runs = mlflow.search_runs()
                logger.info(f"✅ MLflow DB: {len(runs)} runs au total")
                
                if len(runs) < successful_models:
                    logger.warning(
                        f"⚠️ Incohérence MLflow DB: {len(runs)} runs DB "
                        f"vs {successful_models} modèles réussis"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Vérification MLflow DB échouée: {e}")

        # ========================================================================
        # PHASE 12: NETTOYAGE & RETOUR
        # ========================================================================
        gc.collect()

        # Validation finale des résultats
        for result in results:
            if 'metrics' not in result or not isinstance(result['metrics'], dict):
                logger.error(f"❌ CRITIQUE: metrics invalides pour {result.get('model_name', 'Unknown')}")
                result['metrics'] = {'error': 'Metrics non générées'}
                result['success'] = False

        # Log récapitulatif final
        logger.info(
            f"\n{'='*60}\n"
            f"RÉCAPITULATIF ENTRAÎNEMENT\n"
            f"{'='*60}\n"
            f"Task: {task_type}\n"
            f"Modèles: {successful_models}/{len(model_names)} réussis\n"
            f"Features: {len(feature_list)}\n"
            f"SMOTE: {'✅ Activé' if use_smote else '❌ Désactivé'}\n"
            f"MLflow runs collectés: {len(mlflow_runs)}\n"
            f"MLflow runs synchronisés: {total_synchronized if mlflow_runs else 0}\n"
            f"Temps total: {total_time:.2f}s\n"
            f"{'='*60}\n"
        )

        return results

# ===============================================
# FONCTION DE NETTOYAGE DU DOSSIER DES MODÈLES
# ===============================================
def cleanup_models_directory(max_files: int = None):
    """Nettoie le dossier des modèles pour éviter l'accumulation."""
    if max_files is None:
        max_files = TRAINING_CONSTANTS.get("MAX_MODEL_FILES", 50)
    
    try:
        if not os.path.exists("models_output"):
            return
            
        model_files = []
        for filename in os.listdir("models_output"):
            if filename.endswith('.joblib'):
                filepath = os.path.join("models_output", filename)
                model_files.append((filepath, os.path.getctime(filepath)))
        
        model_files.sort(key=lambda x: x[1])
        
        if len(model_files) > max_files:
            for i in range(len(model_files) - max_files):
                filepath, _ = model_files[i]
                os.remove(filepath)
                logger.info(f"🗑️ Fichier modèle supprimé: {filepath}")
                
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage dossier modèles: {e}")

# Nettoyage automatique au chargement du module
cleanup_models_directory()

# Export des fonctions principales
__all__ = [
    'TrainingMonitor',
    'TrainingStateManager',
    'MLflowRunCollector',
    'create_leak_free_pipeline',
    'train_single_model_supervised',
    'train_single_model_unsupervised',
    'train_models',
    'cleanup_models_directory',
    'is_mlflow_available',
    'MLFLOW_AVAILABLE',
    'TRAINING_STATE'
]