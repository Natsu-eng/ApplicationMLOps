"""
🎯 ML Training Orchestrator - Logique Métier Centralisée
Orchestrateur principal pour l'entraînement de modèles ML 
Avec validation renforcée de la feature_list.
Garantit la cohérence et l'intégrité des features avant l'entraînement.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import concurrent.futures
import gc

from src.models.training import (
    create_leak_free_pipeline,
    train_single_model_supervised,
    train_single_model_unsupervised,
    FeatureListValidator
)
from src.data.data_analysis import auto_detect_column_types, detect_imbalance
from src.shared.logging import get_logger
from src.shared.logging import StructuredLogger
from src.config.constants import TRAINING_CONSTANTS, VALIDATION_CONSTANTS

from helpers.data_validators import DataValidator
from monitoring.state_managers import init, STATE  
from helpers.task_detection import safe_get_task_type
from utils.errors_handlers import safe_train_models
from utils.system_utils import check_system_resources
from monitoring.mlflow_collector import get_mlflow_collector

# Intégration MLflow
try:
    import mlflow # type: ignore
    import mlflow.sklearn # type: ignore
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

# Initialisation du state manager (singleton, ne pas réassigner)
STATE = STATE  # Utiliser l'instance importée directement
logger = StructuredLogger(__name__) # pour le logging structuré



# =================================================
# VALIDATEUR DE FEATURE_LIST POUR ORCHESTRATEUR
# =================================================
class FeatureListOrchestrator:
    """
    Orchestrateur spécifique pour la validation de feature_list.
    Garantit la cohérence avant la création du contexte.
    """
    
    @staticmethod
    def validate_from_context(
        df: pd.DataFrame,
        target_column: Optional[str],
        feature_list: List[str],
        task_type: str
    ) -> List[str]:
        """
        Valide et nettoie feature_list depuis le contexte.
        
        Returns:
            feature_list validée et nettoyée
        """
        # VALIDATION basique
        if not feature_list or len(feature_list) == 0:
            logger.error("❌ feature_list vide dans le contexte!")
            
            # Récupération automatique selon task_type
            if task_type == 'clustering':
                feature_list = df.select_dtypes(include=['number']).columns.tolist()
                logger.warning(f"⚠️ feature_list récupérée (clustering): {len(feature_list)} features numériques")
            else:
                feature_list = [col for col in df.columns if col != target_column]
                logger.warning(f"⚠️ feature_list récupérée (supervisé): {len(feature_list)} features")
            
            if not feature_list:
                raise ValueError("Impossible de déterminer feature_list automatiquement")
        
        # VÉRIFICATION existence dans DataFrame
        missing_features = [f for f in feature_list if f not in df.columns]
        if missing_features:
            logger.error(f"❌ Features manquantes dans DataFrame: {missing_features[:5]}")
            feature_list = [f for f in feature_list if f in df.columns]
            logger.warning(f"⚠️ feature_list nettoyée: {len(feature_list)} features restantes")
        
        if not feature_list:
            raise ValueError("Aucune feature valide après nettoyage")
        
        # EXCLUSION target_column (data leakage)
        if task_type in ['classification', 'regression'] and target_column in feature_list:
            feature_list = [f for f in feature_list if f != target_column]
            logger.warning(f"⚠️ target_column '{target_column}' retirée de feature_list")
        
        # VALIDATION finale
        if len(feature_list) == 0:
            raise ValueError("feature_list vide après validation")
        
        logger.info(f"✅ feature_list validée: {len(feature_list)} features")
        logger.debug(f"   Features: {feature_list[:10]}...")
        
        return feature_list


# =========================================
# DATACLASSES POUR CONFIGURATION STRUCTURÉE
# =========================================

@dataclass
class MLTrainingContext:
    """Contexte complet pour un entraînement ML"""
    df: pd.DataFrame
    target_column: Optional[str]
    feature_list: List[str]
    task_type: str
    test_size: float = 0.2
    model_names: List[str] = None
    optimize_hyperparams: bool = False
    preprocessing_config: Dict[str, Any] = None
    use_smote: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_names is None:
            self.model_names = []
        
        if self.preprocessing_config is None:
            self.preprocessing_config = self._get_default_preprocessing()
        
        if self.metadata is None:
            self.metadata = {}
        
        self._validate()
    
    def _get_default_preprocessing(self) -> Dict[str, Any]:
        return {
            'numeric_imputation': 'mean',
            'categorical_imputation': 'most_frequent',
            'remove_constant_cols': True,
            'remove_identifier_cols': True,
            'scale_features': True,
            'scaling_method': 'standard',
            'encoding_method': 'onehot',
            'pca_preprocessing': False,
            'use_smote': self.use_smote,
            'smote_k_neighbors': 5,
            'smote_sampling_strategy': 'auto'
        }
    
    def _validate(self):
        """
        Validation avec FeatureListOrchestrator.
        """
        # Validation DataFrame
        if self.df is None or len(self.df) == 0:
            raise ValueError("DataFrame vide ou None")
        
        # Validation task type
        valid_tasks = ['classification', 'regression', 'clustering']
        if self.task_type not in valid_tasks:
            raise ValueError(f"task_type doit être dans {valid_tasks}, reçu: {self.task_type}")
        
        # Validation target pour supervisé
        if self.task_type in ['classification', 'regression']:
            if not self.target_column:
                raise ValueError(f"target_column requis pour {self.task_type}")
            
            if self.target_column not in self.df.columns:
                raise ValueError(f"target_column '{self.target_column}' n'existe pas dans le DataFrame")
        
        # VALIDATION feature_list via orchestrateur
        try:
            self.feature_list = FeatureListOrchestrator.validate_from_context(
                df=self.df,
                target_column=self.target_column,
                feature_list=self.feature_list,
                task_type=self.task_type
            )
        except ValueError as e:
            logger.error(f"❌ Validation feature_list échouée: {e}")
            raise
        
        # Validation modèles
        if not self.model_names or len(self.model_names) == 0:
            raise ValueError("Au moins un modèle doit être sélectionné")
        
        # Validation test_size
        if not (0 < self.test_size < 1):
            raise ValueError(f"test_size doit être entre 0 et 1, reçu: {self.test_size}")


@dataclass
class MLTrainingResult:
    """Résultat structuré d'un entraînement ML"""
    success: bool
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    training_time: float
    metadata: Dict[str, Any]
    
    @property
    def successful_models(self) -> List[Dict[str, Any]]:
        return [r for r in self.results if r.get('success', False)]
    
    @property
    def failed_models(self) -> List[Dict[str, Any]]:
        return [r for r in self.results if not r.get('success', False)]
    
    @property
    def best_model(self) -> Optional[Dict[str, Any]]:
        if not self.successful_models:
            return None
        
        task_type = self.metadata.get('task_type', 'classification')
        metric_key = {
            'classification': 'accuracy',
            'regression': 'r2',
            'clustering': 'silhouette_score'
        }.get(task_type, 'accuracy')
        
        successful_models = self.successful_models
        if not successful_models:
            return None
            
        return max(
            successful_models,
            key=lambda x: x.get('metrics', {}).get(metric_key, -float('inf'))
        )


# ===========================
# 🎯 ORCHESTRATEUR PRINCIPAL
# ===========================

class MLTrainingOrchestrator:
    """
    Orchestrateur centralisé avec collecteur MLflow intégré.
    """
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
        
        # 🆕 Intégration collecteur MLflow
        self.mlflow_collector = get_mlflow_collector()
        
        # Enregistrement callback vers STATE
        self.mlflow_collector.register_callback(self._sync_run_to_state)
    
    def _sync_run_to_state(self, run: Dict[str, Any]):
        """
        Callback: synchronise chaque run vers STATE immédiatement.
        Appelé automatiquement par le collecteur à chaque nouveau run.
        """
        try:
            from monitoring.state_managers import STATE
            STATE.add_mlflow_run(run)
            logger.debug(f"Callback: Run {run.get('run_id', 'N/A')[:8]} → STATE")
        except Exception as e:
            logger.error(f"❌ Callback sync STATE: {e}")



    def _sync_mlflow_multi_level(self, valid_runs: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Synchronisation atomique multi-niveaux des runs MLflow.  
        Args:
            valid_runs: Liste de runs MLflow validés      
        Returns:
            Dict avec compteurs par source
        """
        sync_counters = {
            'session_state': 0,
            'STATE': 0,
            'STATE.training': 0,
            'total_synchronized': 0
        }
        
        # NIVEAU 1: session_state.mlflow_runs
        try:
            import streamlit as st
            
            if not hasattr(st.session_state, 'mlflow_runs'):
                st.session_state.mlflow_runs = []
            
            existing_ids = {
                r.get('run_id') 
                for r in st.session_state.mlflow_runs 
                if isinstance(r, dict) and r.get('run_id')
            }
            
            new_runs = [
                r for r in valid_runs 
                if r.get('run_id') not in existing_ids
            ]
            
            if new_runs:
                st.session_state.mlflow_runs.extend(new_runs)
                sync_counters['session_state'] = len(new_runs)
                logger.info(
                    f"✅ {len(new_runs)} runs → session_state "
                    f"(total: {len(st.session_state.mlflow_runs)})"
                )
        except Exception as e:
            logger.error(f"❌ Sync session_state: {e}")
        
        # NIVEAU 2: STATE.mlflow_runs
        try:
            from monitoring.state_managers import STATE
            
            existing_ids = {
                r.get('run_id') 
                for r in STATE.mlflow_runs 
                if isinstance(r, dict) and r.get('run_id')
            }
            
            new_runs = [
                r for r in valid_runs 
                if r.get('run_id') not in existing_ids
            ]
            
            if new_runs:
                for run in new_runs:
                    STATE.add_mlflow_run(run)
                
                sync_counters['STATE'] = len(new_runs)
                logger.info(
                    f"✅ {len(new_runs)} runs → STATE "
                    f"(total: {len(STATE.mlflow_runs)})"
                )
        except Exception as e:
            logger.error(f"❌ Sync STATE: {e}")
        
        # NIVEAU 3: STATE.training.mlflow_runs
        try:
            if hasattr(st.session_state, 'training'):
                if not hasattr(st.session_state.training, 'mlflow_runs'):
                    st.session_state.training.mlflow_runs = []
                
                existing_ids = {
                    r.get('run_id') 
                    for r in st.session_state.training.mlflow_runs 
                    if isinstance(r, dict) and r.get('run_id')
                }
                
                new_runs = [
                    r for r in valid_runs 
                    if r.get('run_id') not in existing_ids
                ]
                
                if new_runs:
                    st.session_state.training.mlflow_runs.extend(new_runs)
                    sync_counters['STATE.training'] = len(new_runs)
                    logger.info(
                        f"✅ {len(new_runs)} runs → STATE.training "
                        f"(total: {len(st.session_state.training.mlflow_runs)})"
                    )
        except Exception as e:
            logger.error(f"❌ Sync STATE.training: {e}")
        
        # TOTAL
        sync_counters['total_synchronized'] = sum([
            sync_counters['session_state'],
            sync_counters['STATE'],
            sync_counters['STATE.training']
        ])
        
        return sync_counters
    
    
    def train(self, context: MLTrainingContext) -> MLTrainingResult:
        """
        Orchestration complète de l'entraînement ML avec validation renforcée.
      
        """
        start_time = time.time()
        errors = []
        warnings = []
        results = []
        
        try:
            self.logger.info("=== Début entraînement ML ===", task_type=context.task_type)
            
            # VALIDATION initiale (feature_list déjà validée dans __post_init__)
            validation_result = self._validate_context(context)
            if not validation_result['is_valid']:
                return MLTrainingResult(
                    success=False,
                    results=[],
                    summary={},
                    errors=validation_result['issues'],
                    warnings=validation_result['warnings'],
                    training_time=time.time() - start_time,
                    metadata={'context': context}
                )
            
            warnings.extend(validation_result['warnings'])
            
            # PRÉPARATION des données
            self.logger.info("🔧 Préparation des données")
            data_prep_result = self._prepare_data(context)
            
            if not data_prep_result['success']:
                return MLTrainingResult(
                    success=False,
                    results=[],
                    summary={},
                    errors=[data_prep_result['error']],
                    warnings=warnings,
                    training_time=time.time() - start_time,
                    metadata={'context': context}
                )
            
            # VÉRIFICATION des ressources
            resource_check = check_system_resources(context.df, len(context.model_names))
            if not resource_check['has_enough_resources']:
                errors.extend(resource_check['issues'])
                return MLTrainingResult(
                    success=False,
                    results=[],
                    summary={},
                    errors=errors,
                    warnings=warnings,
                    training_time=time.time() - start_time,
                    metadata={'context': context, 'resource_check': resource_check}
                )
            
            warnings.extend(resource_check['warnings'])
            
            # ENTRAÎNEMENT des modèles
            self.logger.info(f"🤖 Entraînement de {len(context.model_names)} modèles")
            results = self._train_models(context, data_prep_result)
            
            # ANALYSE des résultats
            summary = self._analyze_results(results, context)
            
            # NETTOYAGE mémoire
            gc.collect()
            
            training_time = time.time() - start_time

            try:
                # Récupération depuis collecteur global (source unique de vérité)
                mlflow_runs_collected = self.mlflow_collector.get_runs()
                
                logger.info(f"📊 Runs collectés dans collecteur: {len(mlflow_runs_collected)}")
                
                if mlflow_runs_collected:
                    logger.info(f"🔄 Synchronisation finale de {len(mlflow_runs_collected)} runs...")
                    
                    # Validation des runs
                    valid_runs = [
                        run for run in mlflow_runs_collected 
                        if isinstance(run, dict) and run.get('run_id')
                    ]
                    
                    if not valid_runs:
                        logger.error("❌ Aucun run MLflow valide dans le collecteur")
                    else:
                        logger.info(f"✅ {len(valid_runs)} runs valides à synchroniser")
                        
                        # Synchronisation multi-niveaux
                        sync_counters = self._sync_mlflow_multi_level(valid_runs)
                        
                        logger.info(
                            f"✅ Synchronisation MLflow terminée:\n"
                            f"   • session_state: {sync_counters['session_state']}\n"
                            f"   • STATE: {sync_counters['STATE']}\n"
                            f"   • STATE.training: {sync_counters['STATE.training']}\n"
                            f"   • Total: {sync_counters['total_synchronized']}"
                        )
                
                else:
                    logger.warning("⚠️ Aucun run MLflow dans le collecteur")

            except Exception as sync_error:
                logger.error(f"❌ Erreur synchronisation MLflow: {sync_error}", exc_info=True)
                warnings.append(f"Synchronisation MLflow échouée: {str(sync_error)}")
         
            return MLTrainingResult(
                success=len([r for r in results if r.get('success', False)]) > 0,
                results=results,
                summary=summary,
                errors=errors,
                warnings=warnings,
                training_time=training_time,
                metadata={
                    'context': context,
                    'mlflow_runs_synchronized': len(mlflow_runs_collected) if mlflow_runs_collected else 0,
                    'timestamp': time.time()
                }
            )
        
        except Exception as e:
            self.logger.error(f"❌ Erreur critique orchestration: {e}", exc_info=True)
            return MLTrainingResult(
                success=False,
                results=results,
                summary={},
                errors=[str(e)],
                warnings=warnings,
                training_time=time.time() - start_time,
                metadata={'context': context}
            )
        
        
    def _validate_context(self, context: MLTrainingContext) -> Dict[str, Any]:
        """Validation approfondie du contexte d'entraînement."""
        issues = []
        warnings = []
        
        df_validation = DataValidator.validate_dataframe_for_ml(context.df)
        if not df_validation['is_valid']:
            issues.extend(df_validation['issues'])
        warnings.extend(df_validation['warnings'])
        
        feature_validation = DataValidator.validate_features(
            context.df,
            context.feature_list,
            context.target_column
        )
        if not feature_validation['is_valid']:
            issues.extend(feature_validation['issues'])
        warnings.extend(feature_validation['warnings'])
        
        if context.task_type in ['classification', 'regression']:
            task_info = safe_get_task_type(context.df, context.target_column)
            if task_info['error']:
                issues.append(task_info['error'])
            warnings.extend(task_info.get('warnings', []))
        
        if context.task_type == 'clustering':
            cluster_validation = DataValidator.validate_clustering_features(
                context.df,
                context.feature_list
            )
            if not cluster_validation['valid_features']:
                issues.append("Aucune feature valide pour le clustering")
            warnings.extend(cluster_validation['warnings'])
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def _prepare_data(self, context: MLTrainingContext) -> Dict[str, Any]:
        """
        Préparation données avec validation feature_list renforcée.
        """
        try:
            # feature_list est déjà validée dans le contexte
            logger.info(f"✅ Utilisation de {len(context.feature_list)} features validées")
            
            # Sélection des features
            X = context.df[context.feature_list].copy()
            
            if context.task_type == 'clustering':
                # Vérification colonnes numériques
                non_numeric = X.select_dtypes(exclude=['number']).columns.tolist()
                if non_numeric:
                    self.logger.warning(f"⚠️ Colonnes non-numériques en clustering: {non_numeric[:5]}")
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    for col in non_numeric:
                        X[col] = le.fit_transform(X[col].astype(str))
                    self.logger.info(f"✅ Encodage automatique de {len(non_numeric)} colonnes")
                
                return {
                    'success': True,
                    'X': X,
                    'y': None,
                    'X_train': None,
                    'y_train': None,
                    'X_test': None,
                    'y_test': None,
                    'split_info': {
                        'type': 'no_split',
                        'reason': 'clustering',
                        'n_samples': len(X),
                        'n_features': len(context.feature_list)
                    }
                }
            
            # Supervisé: split train/test
            y = context.df[context.target_column].copy()
            
            # Nettoyage valeurs manquantes dans y
            if y.isnull().any():
                n_missing = y.isnull().sum()
                self.logger.warning(f"⚠️ {n_missing} valeurs manquantes dans target, suppression")
                valid_idx = ~y.isnull()
                X = X[valid_idx]
                y = y[valid_idx]
            
            # Stratification pour classification
            stratify = None
            if context.task_type == 'classification':
                class_counts = y.value_counts()
                min_samples_per_class = class_counts.min()
                
                if min_samples_per_class < 2:
                    self.logger.warning(f"⚠️ Stratification désactivée: {min_samples_per_class} échantillon(s)")
                else:
                    stratify = y
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=context.test_size,
                    random_state=42,
                    stratify=stratify
                )
            except ValueError as e:
                if stratify is not None:
                    self.logger.warning(f"⚠️ Stratification échouée: {e}")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=context.test_size,
                        random_state=42,
                        stratify=None
                    )
                else:
                    raise
            
            self.logger.info(
                "✅ Split effectué",
                train_size=len(X_train),
                test_size=len(X_test),
                stratified=stratify is not None
            )
            
            return {
                'success': True,
                'X': X,
                'y': y,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'split_info': {
                    'type': 'train_test_split',
                    'test_size': context.test_size,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'stratified': stratify is not None,
                    'n_features': len(context.feature_list)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erreur préparation données: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    
    def _train_models(
        self,
        context: MLTrainingContext,
        data_prep: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        - Validation SMOTE AVANT entraînement
        - Synchronisation MLflow garantie
        - Logs structurés
        """
        results = []
        n_models = len(context.model_names)
        
        column_types = auto_detect_column_types(context.df)
        
        # INJECTION feature_list dans preprocessing_choices
        preprocessing_choices_enriched = context.preprocessing_config.copy()
        preprocessing_choices_enriched['feature_list'] = context.feature_list
        
        # ======================================
        # VALIDATION SMOTE PRÉCOCE (AVANT BOUCLE)
        # ======================================
        use_smote_validated = False
        smote_validation_details = {
            'requested': False,
            'from_config': False,
            'from_context': False,
            'validated': False,
            'reason': None
        }
        
        # Récupération SMOTE depuis imbalance_config (prioritaire)
        if hasattr(STATE, 'imbalance_config') and isinstance(STATE.imbalance_config, dict):
            use_smote_from_config = STATE.imbalance_config.get('use_smote', False)
            smote_validation_details['from_config'] = use_smote_from_config
            
            if use_smote_from_config:
                smote_validation_details['requested'] = True
                preprocessing_choices_enriched['use_smote'] = True
                preprocessing_choices_enriched['smote_k_neighbors'] = STATE.imbalance_config.get('smote_k_neighbors', 5)
                preprocessing_choices_enriched['smote_sampling_strategy'] = STATE.imbalance_config.get('smote_sampling_strategy', 'auto')
                
                logger.info(f"✅ SMOTE activé depuis imbalance_config (k={STATE.imbalance_config.get('smote_k_neighbors', 5)})")
        
        # Fallback sur context
        if context.use_smote:
            smote_validation_details['from_context'] = True
            smote_validation_details['requested'] = True
        
        final_use_smote = smote_validation_details['from_config'] or smote_validation_details['from_context']
        
        # VALIDATION si SMOTE demandé
        if final_use_smote and context.task_type == 'classification':
            if context.target_column and data_prep.get('y_train') is not None:
                y_train = data_prep['y_train']
                class_counts = y_train.value_counts()
                min_class_count = class_counts.min()
                smote_k = preprocessing_choices_enriched.get('smote_k_neighbors', 5)
                
                if min_class_count <= smote_k:
                    logger.error(
                        f"❌ SMOTE VALIDATION ÉCHOUÉE!\n"
                        f"   Classe minoritaire: {min_class_count} échantillons\n"
                        f"   k_neighbors: {smote_k}\n"
                        f"   → SMOTE sera DÉSACTIVÉ pour TOUS les modèles"
                    )
                    use_smote_validated = False
                    smote_validation_details['validated'] = False
                    smote_validation_details['reason'] = f"min_class_count ({min_class_count}) ≤ k ({smote_k})"
                    
                    # Désactivation
                    preprocessing_choices_enriched['use_smote'] = False
                    final_use_smote = False
                    
                    # Injection warning global
                    if not hasattr(self, '_global_warnings'):
                        self._global_warnings = []
                    self._global_warnings.append(
                        f"SMOTE désactivé automatiquement: classe minoritaire trop petite "
                        f"({min_class_count} ≤ k={smote_k})"
                    )
                else:
                    use_smote_validated = True
                    smote_validation_details['validated'] = True
                    smote_validation_details['reason'] = f"Validation OK (min_class={min_class_count} > k={smote_k})"
                    logger.info(f"✅ SMOTE validé globalement: {smote_validation_details['reason']}")
            else:
                logger.warning("⚠️ Impossible de valider SMOTE: y_train non disponible")
                final_use_smote = False
        
        # Log récapitulatif SMOTE
        logger.info(
            f"📊 Configuration SMOTE:\n"
            f"   • Demandé: {smote_validation_details['requested']}\n"
            f"   • Source config: {smote_validation_details['from_config']}\n"
            f"   • Source context: {smote_validation_details['from_context']}\n"
            f"   • Validé: {smote_validation_details['validated']}\n"
            f"   • Raison: {smote_validation_details['reason']}\n"
            f"   • Final: {'✅ ACTIVÉ' if final_use_smote else '❌ DÉSACTIVÉ'}"
        )
        
        # ========================================================================
        # FILTRAGE column_types selon feature_list
        # ========================================================================
        feature_set = set(context.feature_list)
        filtered_column_types = {}
        
        for col_type, cols in column_types.items():
            valid_cols = [col for col in cols if col in feature_set]
            if valid_cols:
                filtered_column_types[col_type] = valid_cols
        
        logger.info("✅ Column types filtrés:")
        for col_type, cols in filtered_column_types.items():
            logger.info(f"   • {col_type}: {len(cols)} colonnes")
        
        total_cols = sum(len(cols) for cols in filtered_column_types.values())
        if total_cols == 0:
            logger.error("❌ Aucune colonne valide après filtrage!")
            return [{
                'model_name': 'Validation Error',
                'success': False,
                'metrics': {'error': 'Aucune colonne valide pour preprocessing'},
                'training_time': 0,
                'warnings': ['Filtrage column_types a tout supprimé']
            }]

        # ========================================================================
        # WRAPPER D'ENTRAÎNEMENT CORRIGÉ (avec MLflow)
        # ========================================================================
        def train_single_model_wrapper(model_name: str) -> Dict[str, Any]:
            """
            Wrapper CORRIGÉ avec intégration train_single_model_with_mlflow.
            """
            try:
                self.logger.info(f"🔧 Entraînement: {model_name}")
                
                # 🎯 IMPORT DE LA FONCTION AVEC MLFLOW
                from src.models.training import train_single_model_with_mlflow
                from monitoring.performance_monitor import TrainingMonitor
                from utils.mlflow import get_git_info
                
                # Préparation des données selon task_type
                if context.task_type == 'clustering':
                    X_for_training = data_prep['X'][context.feature_list].copy()
                    X_train = None
                    y_train = None
                    X_test = None
                    y_test = None
                    X = X_for_training
                else:
                    X_train = data_prep['X_train'][context.feature_list].copy()
                    X_test = data_prep['X_test'][context.feature_list].copy()
                    y_train = data_prep['y_train']
                    y_test = data_prep['y_test']
                    X = None
                
                # Création monitor
                monitor = TrainingMonitor()
                
                # Récupération git info pour MLflow
                git_info = get_git_info() if MLFLOW_AVAILABLE else {}
                
                # 🎯 APPEL FONCTION AVEC MLFLOW (retourne result ET mlflow_run_data)
                result, mlflow_run_data = train_single_model_with_mlflow(
                    model_name=model_name,
                    task_type=context.task_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    X=X,
                    column_types=filtered_column_types,
                    preprocessing_choices=preprocessing_choices_enriched,
                    use_smote=final_use_smote,
                    optimize=context.optimize_hyperparams,
                    feature_list=context.feature_list,
                    git_info=git_info,
                    label_encoder=None,
                    sample_metrics=True,
                    max_samples_metrics=100000,
                    monitor=monitor,
                    mlflow_enabled=MLFLOW_AVAILABLE
                )
                
                if result is None:
                    return {
                        'model_name': model_name,
                        'success': False,
                        'metrics': {'error': 'Entraînement échoué'},
                        'training_time': 0,
                        'warnings': ['Result None']
                    }
                
                # COLLECTE MLflow_run_data si présent
                if mlflow_run_data and isinstance(mlflow_run_data, dict):
                    # Vérifier si le run n'est pas déjà collecté
                    run_id = mlflow_run_data.get('run_id', 'N/A')[:8]
                    if not self.mlflow_collector.run_exists(mlflow_run_data.get('run_id')):
                        collected = self.mlflow_collector.add_run(mlflow_run_data)
                        if collected:
                            logger.info(f"✅ {model_name}: Run MLflow collecté (ID: {run_id})")
                        else:
                            logger.warning(f"⚠️ {model_name}: Run MLflow non collecté (dupliqué ou invalide)")
                    else:
                        logger.info(f"✅ {model_name}: Run MLflow déjà collecté (ID: {run_id})")
                else:
                    logger.warning(f"⚠️ {model_name}: Aucun mlflow_run_data retourné")
                
                # Enrichissement résultat avec infos contexte
                result['task_type'] = context.task_type
                result['feature_names'] = context.feature_list
                result['smote_applied'] = final_use_smote
                result['smote_validated'] = use_smote_validated
                
                # Ajout données pour évaluation
                if context.task_type == 'clustering':
                    if result.get('labels') is None:
                        logger.error(f"❌ {model_name}: Labels clustering manquants!")
                    else:
                        logger.info(f"✅ {model_name}: {len(result['labels'])} labels clustering")
                else:
                    result['X_train'] = X_train
                    result['y_train'] = y_train
                    result['X_test'] = X_test
                    result['y_test'] = y_test
                    
                    if len(X_test) > 1000:
                        result['X_sample'] = X_test.iloc[:1000].copy()
                        result['y_sample'] = y_test.iloc[:1000].copy()
                
                return result
                
            except Exception as e:
                self.logger.error(f"❌ Erreur {model_name}: {e}", exc_info=True)
                return {
                    'model_name': model_name,
                    'success': False,
                    'metrics': {'error': str(e)},
                    'training_time': 0,
                    'task_type': context.task_type,
                    'feature_names': context.feature_list,
                    'warnings': [str(e)]
                }
        
        # ===================================
        # EXÉCUTION (parallèle/séquentielle)
        # ===================================
        n_jobs = TRAINING_CONSTANTS.get("N_JOBS", -1)
        
        if n_jobs == 1 or n_models == 1:
            self.logger.info("Exécution séquentielle")
            results = [train_single_model_wrapper(name) for name in context.model_names]
        else:
            self.logger.info(f"Exécution parallèle (n_jobs={n_jobs})")
            max_workers = min(n_models, abs(n_jobs)) if n_jobs > 0 else n_models
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_model = {
                    executor.submit(train_single_model_wrapper, name): name
                    for name in context.model_names
                }
                
                for future in concurrent.futures.as_completed(future_to_model):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        model_name = future_to_model[future]
                        self.logger.error(f"❌ Future failed {model_name}: {e}")
                        results.append({
                            'model_name': model_name,
                            'success': False,
                            'metrics': {'error': str(e)},
                            'training_time': 0,
                            'warnings': [str(e)]
                        })
        
        # ============================
        # DIAGNOSTIC FINAL (CRITIQUE)
        # ============================
        successful_models = [r for r in results if r.get('success', False)]
        
        logger.info(f"\n{'='*60}\n📊 DIAGNOSTIC POST-ENTRAÎNEMENT\n{'='*60}")
        
        for model in successful_models:
            model_name = model.get('model_name', 'Unknown')
            
            if context.task_type == 'clustering':
                has_labels = model.get('labels') is not None
                has_X = model.get('X_train') is not None
                
                if has_labels and has_X:
                    n_labels = len(model['labels']) if hasattr(model['labels'], '__len__') else 'N/A'
                    logger.info(f"✅ {model_name}: Données clustering COMPLÈTES (labels: {n_labels})")
                else:
                    logger.error(f"❌ {model_name}: DONNÉES CLUSTERING MANQUANTES")
                    logger.error(f"   has_labels: {has_labels}, has_X: {has_X}")
            else:
                has_test_data = model.get('X_test') is not None and model.get('y_test') is not None
                smote_applied = model.get('smote_applied', False)
                smote_validated = model.get('smote_validated', False)
                
                if has_test_data:
                    smote_status = f"SMOTE: {'✅ appliqué' if smote_applied else '❌ désactivé'}"
                    if smote_applied and not smote_validated:
                        smote_status += " (non validé)"
                    logger.info(f"✅ {model_name}: Données test OK ({smote_status})")
                else:
                    logger.error(f"❌ {model_name}: DONNÉES TEST MANQUANTES")
        
        logger.info(f"{'='*60}\n")
        
        return results
    
    
    def _analyze_results(
        self,
        results: List[Dict[str, Any]],
        context: MLTrainingContext
    ) -> Dict[str, Any]:
        """Analyse résultats et sélection meilleur modèle"""
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        metric_key = {
            'classification': 'accuracy',
            'regression': 'r2',
            'clustering': 'silhouette_score'
        }.get(context.task_type, 'accuracy')
        
        models_with_metrics = []
        models_without_metrics = []
        
        for model in successful:
            model_name = model.get('model_name', 'Unknown')
            
            if 'metrics' not in model:
                self.logger.error(f"❌ {model_name}: 'metrics' manquante!")
                models_without_metrics.append(model_name)
                continue
            
            metrics = model.get('metrics', {})
            
            if not isinstance(metrics, dict):
                self.logger.error(f"❌ {model_name}: 'metrics' n'est pas un dict!")
                models_without_metrics.append(model_name)
                continue
            
            if metric_key not in metrics:
                self.logger.warning(f"⚠️ {model_name}: métrique '{metric_key}' manquante")
                models_without_metrics.append(model_name)
                continue
            
            metric_value = metrics[metric_key]
            
            if not isinstance(metric_value, (int, float, np.number)):
                self.logger.warning(f"⚠️ {model_name}: métrique invalide: {metric_value}")
                models_without_metrics.append(model_name)
                continue
            
            if np.isnan(metric_value) or np.isinf(metric_value):
                self.logger.warning(f"⚠️ {model_name}: métrique NaN/Inf: {metric_value}")
                models_without_metrics.append(model_name)
                continue
            
            models_with_metrics.append(model)
        
        self.logger.info(f"📊 Analyse résultats:")
        self.logger.info(f"   • Total: {len(results)}")
        self.logger.info(f"   • Réussis: {len(successful)}")
        self.logger.info(f"   • Échoués: {len(failed)}")
        self.logger.info(f"   • Avec métriques: {len(models_with_metrics)}")
        self.logger.info(f"   • Sans métriques: {len(models_without_metrics)}")
        
        if models_without_metrics:
            self.logger.error(f"⚠️ Modèles sans métriques: {models_without_metrics}")
        
        best_model = None
        best_score = None
        
        if models_with_metrics:
            try:
                best_model = max(
                    models_with_metrics,
                    key=lambda x: float(x['metrics'][metric_key])
                )
                best_score = float(best_model['metrics'][metric_key])
                
                self.logger.info(f"✅ Meilleur modèle: {best_model['model_name']} ({metric_key}={best_score:.4f})")
                
            except Exception as e:
                self.logger.error(f"❌ Erreur sélection meilleur modèle: {e}", exc_info=True)
        else:
            self.logger.error("❌ Aucun modèle avec métriques valides")
        
        summary = {
            'total_models': len(results),
            'successful_models': len(successful),
            'failed_models': len(failed),
            'models_with_valid_metrics': len(models_with_metrics),
            'models_without_metrics': models_without_metrics,
            'best_model': best_model['model_name'] if best_model else None,
            'best_score': best_score,
            'metric_used': metric_key,
            'all_models': [r.get('model_name', 'Unknown') for r in results],
            'successful_model_names': [r.get('model_name', 'Unknown') for r in successful],
            'failed_model_names': [r.get('model_name', 'Unknown') for r in failed],
            'recommendations': self._generate_recommendations(results, context)
        }
        
        if models_without_metrics:
            summary['debug_info'] = {
                'models_without_metrics_details': []
            }
            
            for model in successful:
                model_name = model.get('model_name', 'Unknown')
                if model_name in models_without_metrics:
                    debug_entry = {
                        'model_name': model_name,
                        'has_metrics_key': 'metrics' in model,
                        'metrics_type': str(type(model.get('metrics'))),
                        'available_keys': list(model.keys())
                    }
                    
                    if 'metrics' in model:
                        debug_entry['metrics_keys'] = list(model['metrics'].keys()) if isinstance(model['metrics'], dict) else 'Not a dict'
                    
                    summary['debug_info']['models_without_metrics_details'].append(debug_entry)
        
        return summary
   
    def _generate_recommendations(
        self,
        results: List[Dict[str, Any]],
        context: MLTrainingContext
    ) -> List[str]:
        """Génère recommandations basées sur les résultats"""
        recommendations = []
        
        successful = [r for r in results if r.get('success', False)]
        
        if not successful:
            recommendations.append("❌ Aucun modèle entraîné. Vérifiez la qualité des données.")
            return recommendations
        
        metric_key = {
            'classification': 'accuracy',
            'regression': 'r2',
            'clustering': 'silhouette_score'
        }.get(context.task_type, 'accuracy')
        
        scores = [r['metrics'].get(metric_key, 0) for r in successful]
        avg_score = np.mean(scores) if scores else 0
        
        if context.task_type == 'classification':
            if avg_score < 0.7:
                recommendations.append("⚠️ Scores faibles (<70%). Suggestions: plus de données, feature engineering, SMOTE.")
            elif avg_score > 0.95:
                recommendations.append("⚠️ Scores très élevés (>95%). Vérifiez le data leakage.")
        
        elif context.task_type == 'regression':
            if avg_score < 0.5:
                recommendations.append("⚠️ R² faible (<0.5). Suggestions: feature engineering, modèles non-linéaires.")
        
        elif context.task_type == 'clustering':
            if avg_score < 0.3:
                recommendations.append("⚠️ Silhouette faible (<0.3). Suggestions: normalisation, réduction dimensionnelle.")
        
        if not context.optimize_hyperparams:
            recommendations.append("💡 Activez l'optimisation des hyperparamètres pour améliorer les performances.")
        
        if context.task_type == 'classification' and context.target_column:
            imbalance_info = detect_imbalance(context.df, context.target_column)
            if imbalance_info.get('is_imbalanced') and not context.use_smote:
                recommendations.append(f"⚠️ Déséquilibre détecté (ratio: {imbalance_info['imbalance_ratio']:.2f}). Activez SMOTE.")
        
        return recommendations


# =================
# INSTANCE GLOBALE
# =================

ml_training_orchestrator = MLTrainingOrchestrator()


# ==============
# WRAPPER LEGACY
# ==============

def train_models_legacy_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """
    Wrapper de compatibilité avec validation feature_list renforcée.
    """
    try:
        # Récupération feature_list avec validation stricte
        feature_list = kwargs.get('feature_list', [])
        
        if not feature_list:
            df = kwargs['df']
            target_column = kwargs.get('target_column')
            
            if target_column:
                feature_list = [col for col in df.columns if col != target_column]
            else:
                feature_list = df.select_dtypes(include=['number']).columns.tolist()
            
            if not feature_list:
                raise ValueError("❌ Impossible de déterminer feature_list automatiquement")
            
            logger.warning(f"⚠️ feature_list vide, récupération auto: {len(feature_list)} features")
        
        context = MLTrainingContext(
            df=kwargs['df'],
            target_column=kwargs.get('target_column'),
            feature_list=feature_list,
            task_type=kwargs.get('task_type', 'classification'),
            test_size=kwargs.get('test_size', 0.2),
            model_names=kwargs.get('model_names', []),
            optimize_hyperparams=kwargs.get('optimize', False),
            preprocessing_config=kwargs.get('preprocessing_choices'),
            use_smote=kwargs.get('use_smote', False),
            metadata=kwargs
        )
        
        result = ml_training_orchestrator.train(context)
        
        return result.results
        
    except Exception as e:
        logger.error(f"❌ Erreur wrapper legacy: {e}", exc_info=True)
        return [{
            'model_name': 'Error',
            'success': False,
            'metrics': {'error': str(e)},
            'warnings': [str(e)],
            'training_time': 0
        }]