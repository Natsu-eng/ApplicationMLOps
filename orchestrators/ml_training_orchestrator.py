"""
🎯 ML Training Orchestrator - Logique Métier Centralisée
Architecture Production pour ML Classique (Tabular Data)
Version: 2.0 | Production-Ready | Fixed Imports
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
    train_single_model_unsupervised
)
from src.data.data_analysis import auto_detect_column_types, detect_imbalance
from src.shared.logging import StructuredLogger
from src.config.constants import TRAINING_CONSTANTS, VALIDATION_CONSTANTS

from helpers.data_validators import DataValidator
from monitoring.state_managers import init, STATE  
from helpers.task_detection import safe_get_task_type
from utils.errors_handlers import safe_train_models
from utils.system_utils import check_system_resources

# Initialisation du state manager
STATE = init()
logger = StructuredLogger(__name__)


# ============================================================================
# DATACLASSES POUR CONFIGURATION STRUCTURÉE
# ============================================================================

@dataclass
class MLTrainingContext:
    """
    Contexte complet pour un entraînement ML classique.
    Remplace les dictionnaires éparpillés.
    """
    # Données
    df: pd.DataFrame
    target_column: Optional[str]
    feature_list: List[str]
    
    # Configuration de la tâche
    task_type: str  # 'classification', 'regression', 'clustering'
    test_size: float = 0.2
    
    # Modèles
    model_names: List[str] = None
    optimize_hyperparams: bool = False
    
    # Preprocessing
    preprocessing_config: Dict[str, Any] = None
    use_smote: bool = False
    
    # Métadonnées
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validation post-initialisation"""
        if self.model_names is None:
            self.model_names = []
        
        if self.preprocessing_config is None:
            self.preprocessing_config = self._get_default_preprocessing()
        
        if self.metadata is None:
            self.metadata = {}
        
        # Validation
        self._validate()
    
    def _get_default_preprocessing(self) -> Dict[str, Any]:
        """Configuration preprocessing par défaut"""
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
        """Validation complète du contexte avec gestion améliorée"""
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
            
            # Vérifier que target existe dans le DataFrame
            if self.target_column not in self.df.columns:
                raise ValueError(f"target_column '{self.target_column}' n'existe pas dans le DataFrame")
        
        # Validation features avec message détaillé
        if not self.feature_list or len(self.feature_list) == 0:
            # Tenter une récupération automatique
            if self.task_type == 'clustering':
                # Pour clustering, utiliser toutes les colonnes numériques
                numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    self.feature_list = numeric_cols
                    logger.warning(f"⚠️ feature_list vide, récupération auto: {len(numeric_cols)} colonnes numériques")
                else:
                    raise ValueError("feature_list vide et aucune colonne numérique détectée pour clustering")
            else:
                # Pour supervisé, utiliser toutes colonnes sauf target
                all_cols = [col for col in self.df.columns if col != self.target_column]
                if all_cols:
                    self.feature_list = all_cols
                    logger.warning(f"⚠️ feature_list vide, récupération auto: {len(all_cols)} colonnes (toutes sauf target)")
                else:
                    raise ValueError("feature_list vide et aucune colonne utilisable détectée")
        
        # Vérifier que les features existent dans le DataFrame
        missing_features = [f for f in self.feature_list if f not in self.df.columns]
        if missing_features:
            raise ValueError(f"Features manquantes dans le DataFrame: {missing_features[:5]}...")
        
        # Éviter que target soit dans feature_list (data leakage)
        if self.task_type in ['classification', 'regression'] and self.target_column in self.feature_list:
            self.feature_list = [f for f in self.feature_list if f != self.target_column]
            logger.warning(f"⚠️ target_column '{self.target_column}' retirée de feature_list (data leakage)")
        
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
        """Filtre les modèles avec succès"""
        return [r for r in self.results if r.get('success', False)]
    
    @property
    def failed_models(self) -> List[Dict[str, Any]]:
        """Filtre les modèles échoués"""
        return [r for r in self.results if not r.get('success', False)]
    
    @property
    def best_model(self) -> Optional[Dict[str, Any]]:
        """Retourne le meilleur modèle selon la métrique principale"""
        if not self.successful_models:
            return None
        
        # Déterminer la métrique de tri
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


# ============================================================================
# ORCHESTRATEUR PRINCIPAL
# ============================================================================

class MLTrainingOrchestrator:
    """
    Orchestrateur centralisé pour l'entraînement ML classique.
    """
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
    
    def train(self, context: MLTrainingContext) -> MLTrainingResult:
        """
        Point d'entrée principal pour l'entraînement.
        """
        start_time = time.time()
        errors = []
        warnings = []
        results = []
        
        try:
            # 1. Validation initiale
            self.logger.info("=== Début entraînement ML ===", task_type=context.task_type)
            
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
            
            # 2. Préparation des données
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
            
            # 3. Vérification des ressources système
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
            
            # 4. Entraînement des modèles
            self.logger.info(f"🤖 Entraînement de {len(context.model_names)} modèles")
            results = self._train_models(context, data_prep_result)
            
            # 5. Analyse des résultats
            summary = self._analyze_results(results, context)
            
            # 6. Nettoyage mémoire
            gc.collect()
            
            training_time = time.time() - start_time
            
            return MLTrainingResult(
                success=len([r for r in results if r.get('success')]) > 0,
                results=results,
                summary=summary,
                errors=errors,
                warnings=warnings,
                training_time=training_time,
                metadata={
                    'context': context,
                    'data_prep': data_prep_result,
                    'resource_check': resource_check
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
        """Validation approfondie du contexte"""
        issues = []
        warnings = []
        
        # Validation DataFrame
        df_validation = DataValidator.validate_dataframe_for_ml(context.df)
        if not df_validation['is_valid']:
            issues.extend(df_validation['issues'])
        warnings.extend(df_validation['warnings'])
        
        # Validation features
        feature_validation = DataValidator.validate_features(
            context.df,
            context.feature_list,
            context.target_column
        )
        if not feature_validation['is_valid']:
            issues.extend(feature_validation['issues'])
        warnings.extend(feature_validation['warnings'])
        
        # Validation spécifique au type de tâche
        if context.task_type in ['classification', 'regression']:
            task_info = safe_get_task_type(context.df, context.target_column)
            if task_info['error']:
                issues.append(task_info['error'])
            warnings.extend(task_info.get('warnings', []))
        
        # Validation clustering
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
        """Prépare les données pour l'entraînement avec gestion OPTIMISÉE"""
        try:
            # VALIDATION ROBUSTE DES FEATURES
            if not context.feature_list:
                # Tentative de récupération depuis différentes sources
                possible_sources = [
                    getattr(context, 'feature_list', None),
                    getattr(context.metadata, 'feature_list', None) if context.metadata else None,
                    [col for col in context.df.columns if col != context.target_column] if context.target_column else context.df.columns.tolist()
                ]
                
                for source in possible_sources:
                    if source and len(source) > 0:
                        context.feature_list = source
                        self.logger.warning(f"🔄 feature_list récupérée automatiquement: {len(source)} features")
                        break
                
                # Dernier recours : toutes les colonnes sauf target
                if not context.feature_list:
                    if context.target_column and context.target_column in context.df.columns:
                        context.feature_list = [col for col in context.df.columns if col != context.target_column]
                    else:
                        context.feature_list = context.df.columns.tolist()
                    
                    self.logger.warning(f"🔄 feature_list définie par défaut: {len(context.feature_list)} features")
            
            # Validation finale
            missing_features = [f for f in context.feature_list if f not in context.df.columns]
            if missing_features:
                self.logger.error(f"❌ Features manquantes: {missing_features}")
                # Nettoyage des features manquantes
                context.feature_list = [f for f in context.feature_list if f in context.df.columns]
                self.logger.warning(f"🔄 Features nettoyées: {len(context.feature_list)} restantes")
            
            if not context.feature_list:
                return {
                    'success': False,
                    'error': "Aucune feature valide après nettoyage"
                }
            
            # SÉLECTION SÉCURISÉE DES FEATURES
            X = context.df[context.feature_list].copy()
            
            if context.task_type == 'clustering':
                # Pour clustering, vérifier que les données sont numériques
                non_numeric = X.select_dtypes(exclude=['number']).columns.tolist()
                if non_numeric:
                    self.logger.warning(f"⚠️ Colonnes non-numériques dans clustering: {non_numeric[:5]}")
                    # Option 1: Encoder automatiquement
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    for col in non_numeric:
                        X[col] = le.fit_transform(X[col].astype(str))
                    self.logger.info(f"✅ Encodage automatique de {len(non_numeric)} colonnes catégorielles")
                
                # Clustering: pas de split
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
            
            # Vérifier que y n'a pas de valeurs manquantes
            if y.isnull().any():
                n_missing = y.isnull().sum()
                self.logger.warning(f"⚠️ {n_missing} valeurs manquantes dans target, suppression des lignes")
                valid_idx = ~y.isnull()
                X = X[valid_idx]
                y = y[valid_idx]
            
            # Stratification pour classification
            stratify = None
            if context.task_type == 'classification':
                # Vérifier qu'il y a assez d'échantillons par classe pour stratifier
                class_counts = y.value_counts()
                min_samples_per_class = class_counts.min()
                
                if min_samples_per_class < 2:
                    self.logger.warning(
                        f"⚠️ Stratification désactivée: classe avec {min_samples_per_class} échantillon(s) seulement"
                    )
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
                # Si stratification échoue, réessayer sans
                if stratify is not None:
                    self.logger.warning(f"⚠️ Stratification échouée, réessai sans: {e}")
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
        Entraîne tous les modèles sélectionnés.
        - Injection systématique de X_test, y_test dans les résultats pour l'évaluation
        - Sauvegarde des données d'entraînement et de test pour les visualisations
        - Gestion robuste de la mémoire avec échantillons réduits
        """
        results = []
        n_models = len(context.model_names)
        
        # Détection auto-détection des types de colonnes
        column_types = auto_detect_column_types(context.df)
        
        # ========================================================================
        # NJECTION feature_list dans preprocessing_choices
        # ========================================================================
        preprocessing_choices_enriched = context.preprocessing_config.copy()
        preprocessing_choices_enriched['feature_list'] = context.feature_list
        
        logger.info(f"✅ feature_list injectée: {len(context.feature_list)} colonnes")
        logger.debug(f"   Détail: {context.feature_list[:10]}...")
        
        # ========================================================================
        # FILTRAGE column_types pour ne garder QUE les colonnes de feature_list
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
        
        # Validation: au moins une colonne doit rester
        total_cols = sum(len(cols) for cols in filtered_column_types.values())
        if total_cols == 0:
            logger.error("❌ Aucune colonne valide après filtrage!")
            return [{
                'model_name': 'Validation Error',
                'success': False,
                'metrics': {'error': 'Aucune colonne valide pour le preprocessing'},
                'training_time': 0,
                'warnings': ['Filtrage column_types a tout supprimé']
            }]
        
        def train_single_model_wrapper(model_name: str) -> Dict[str, Any]:
            """Wrapper pour entraînement d'un modèle avec gestion d'erreurs"""
            try:
                self.logger.info(f"🔧 Entraînement: {model_name}")
                
                # Création du pipeline avec preprocessing_choices ENRICHI
                pipeline, param_grid = create_leak_free_pipeline(
                    model_name=model_name,
                    task_type=context.task_type,
                    column_types=filtered_column_types,  # Utilise les colonnes FILTRÉES
                    preprocessing_choices=preprocessing_choices_enriched,  # Avec feature_list
                    use_smote=context.use_smote,
                    optimize_hyperparams=context.optimize_hyperparams
                )
                
                if pipeline is None:
                    return {
                        'model_name': model_name,
                        'success': False,
                        'metrics': {'error': 'Pipeline creation failed'},
                        'training_time': 0,
                        'warnings': ['Pipeline creation failed']
                    }
                
                # ========================================================================
                # VALIDATION: S'assurer que X_train/X/X_test contiennent SEULEMENT feature_list
                # ========================================================================
                if context.task_type == 'clustering':
                    X_for_training = data_prep['X'][context.feature_list].copy()
                    
                    logger.debug(f"✅ X clustering préparé: {X_for_training.shape}")
                    
                    result = train_single_model_unsupervised(
                        model_name=model_name,
                        pipeline=pipeline,
                        X=X_for_training,
                        param_grid=param_grid,
                        monitor=None
                    )
                    
                    # Injection données clustering pour visualisations
                    result['X_train'] = X_for_training
                    result['labels'] = result.get('predictions')
                    result['feature_names'] = context.feature_list
                    
                else:
                    X_train = data_prep['X_train'][context.feature_list].copy()
                    X_test = data_prep['X_test'][context.feature_list].copy()
                    y_train = data_prep['y_train']
                    y_test = data_prep['y_test']
                    
                    logger.debug(f"✅ X_train préparé: {X_train.shape}, X_test: {X_test.shape}")
                    
                    result = train_single_model_supervised(
                        model_name=model_name,
                        pipeline=pipeline,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        param_grid=param_grid,
                        task_type=context.task_type,
                        monitor=None
                    )
                    
                    # Injection systématique des données pour l'évaluation
                    result['X_train'] = X_train
                    result['y_train'] = y_train
                    result['X_test'] = X_test  # ← CE QUI MANQUAIT !
                    result['y_test'] = y_test  # ← CE QUI MANQUAIT !
                    result['feature_names'] = context.feature_list
                    
                    # Échantillon réduit pour éviter la mémoire excessive
                    if len(X_test) > 1000:
                        result['X_sample'] = X_test.iloc[:1000].copy()
                        result['y_sample'] = y_test.iloc[:1000].copy()
                        logger.info(f"✅ Échantillon réduit créé: 1000/{len(X_test)} échantillons")
                
                # Enrichissement du résultat
                result['task_type'] = context.task_type
                result['feature_names'] = context.feature_list
                
                # Validation que les données sont bien sauvegardées
                data_saved = {
                    'X_train': result.get('X_train') is not None,
                    'X_test': result.get('X_test') is not None,
                    'y_train': result.get('y_train') is not None,
                    'y_test': result.get('y_test') is not None,
                    'labels': result.get('labels') is not None
                }
                
                logger.info(f"✅ Données sauvegardées pour {model_name}: {data_saved}")
                
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
        
        # Exécution parallèle ou séquentielle
        n_jobs = TRAINING_CONSTANTS.get("N_JOBS", -1)
        
        if n_jobs == 1 or n_models == 1:
            # Mode séquentiel
            self.logger.info("Exécution séquentielle")
            results = [train_single_model_wrapper(name) for name in context.model_names]
        else:
            # Mode parallèle
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
        
        # Log final de validation des données sauvegardées
        successful_models = [r for r in results if r.get('success', False)]
        for model in successful_models:
            model_name = model.get('model_name', 'Unknown')
            has_test_data = model.get('X_test') is not None and model.get('y_test') is not None
            has_train_data = model.get('X_train') is not None and model.get('y_train') is not None
            
            if context.task_type != 'clustering':
                if has_test_data:
                    logger.info(f"✅ {model_name}: Données de test sauvegardées ✅")
                else:
                    logger.error(f"❌ {model_name}: DONNÉES DE TEST MANQUANTES ❌")
        
        return results
    
    def _analyze_results(
        self,
        results: List[Dict[str, Any]],
        context: MLTrainingContext
    ) -> Dict[str, Any]:
        """
        Analyse les résultats et génère des recommandations.
        - Validation STRICTE de l'existence de 'metrics'
        - Gestion robuste des modèles sans métriques
        - Logs détaillés pour debugging
        """
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        # Déterminer la métrique principale
        metric_key = {
            'classification': 'accuracy',
            'regression': 'r2',
            'clustering': 'silhouette_score'
        }.get(context.task_type, 'accuracy')
        
        # ========================================================================
        # 🆕 VALIDATION STRICTE: Vérifier que 'metrics' existe
        # ========================================================================
        models_with_metrics = []
        models_without_metrics = []
        
        for model in successful:
            model_name = model.get('model_name', 'Unknown')
            
            # Vérification stricte
            if 'metrics' not in model:
                self.logger.error(f"❌ CRITIQUE: Modèle {model_name} sans clé 'metrics'!")
                self.logger.error(f"   Clés disponibles: {list(model.keys())}")
                models_without_metrics.append(model_name)
                continue
            
            metrics = model.get('metrics', {})
            
            if not isinstance(metrics, dict):
                self.logger.error(f"❌ CRITIQUE: Modèle {model_name} - 'metrics' n'est pas un dict!")
                self.logger.error(f"   Type: {type(metrics)}, Valeur: {metrics}")
                models_without_metrics.append(model_name)
                continue
            
            if metric_key not in metrics:
                self.logger.warning(f"⚠️ Modèle {model_name} sans métrique '{metric_key}'")
                self.logger.warning(f"   Métriques disponibles: {list(metrics.keys())}")
                models_without_metrics.append(model_name)
                continue
            
            metric_value = metrics[metric_key]
            
            if not isinstance(metric_value, (int, float, np.number)):
                self.logger.warning(f"⚠️ Modèle {model_name} - métrique '{metric_key}' invalide: {metric_value}")
                models_without_metrics.append(model_name)
                continue
            
            if np.isnan(metric_value) or np.isinf(metric_value):
                self.logger.warning(f"⚠️ Modèle {model_name} - métrique '{metric_key}' NaN/Inf: {metric_value}")
                models_without_metrics.append(model_name)
                continue
            
            # Modèle valide
            models_with_metrics.append(model)
        
        # ========================================================================
        # LOG DÉTAILLÉ pour debugging
        # ========================================================================
        self.logger.info(f"📊 Analyse des résultats:")
        self.logger.info(f"   • Total modèles: {len(results)}")
        self.logger.info(f"   • Modèles réussis: {len(successful)}")
        self.logger.info(f"   • Modèles échoués: {len(failed)}")
        self.logger.info(f"   • Modèles avec métriques valides: {len(models_with_metrics)}")
        self.logger.info(f"   • Modèles sans métriques: {len(models_without_metrics)}")
        
        if models_without_metrics:
            self.logger.error(f"⚠️ Modèles SANS métriques valides: {models_without_metrics}")
        
        # ========================================================================
        # SÉLECTION DU MEILLEUR MODÈLE
        # ========================================================================
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
            self.logger.error("❌ Aucun modèle avec métriques valides pour sélection du meilleur")
        
        # ========================================================================
        # CONSTRUCTION DU SUMMARY
        # ========================================================================
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
        
        # ========================================================================
        # 🆕 AJOUT D'INFORMATIONS DE DEBUG
        # ========================================================================
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
        """Génère des recommandations basées sur les résultats"""
        recommendations = []
        
        successful = [r for r in results if r.get('success', False)]
        
        if not successful:
            recommendations.append("❌ Aucun modèle n'a pu être entraîné. Vérifiez la qualité des données.")
            return recommendations
        
        # Analyse de performance
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
                recommendations.append("⚠️ Silhouette faible (<0.3). Suggestions: normalisation, réduction dimensionnelle, autre nombre de clusters.")
        
        # Recommandations sur l'optimisation
        if not context.optimize_hyperparams:
            recommendations.append("💡 Activez l'optimisation des hyperparamètres pour améliorer les performances.")
        
        # Recommandations sur le déséquilibre (classification)
        if context.task_type == 'classification' and context.target_column:
            imbalance_info = detect_imbalance(context.df, context.target_column)
            if imbalance_info.get('is_imbalanced') and not context.use_smote:
                recommendations.append(f"⚠️ Déséquilibre détecté (ratio: {imbalance_info['imbalance_ratio']:.2f}). Activez SMOTE.")
        
        return recommendations


# ============================================================================
# INSTANCE GLOBALE
# ============================================================================

ml_training_orchestrator = MLTrainingOrchestrator()


# ============================================================================
# HELPERS POUR COMPATIBILITÉ AVEC CODE EXISTANT
# ============================================================================

def train_models_legacy_wrapper(**kwargs) -> List[Dict[str, Any]]:
    """
    Wrapper de compatibilité pour l'ancien train_models()
    avec gestion robuste de feature_list
    """
    try:
        # Récupération intelligente de feature_list
        feature_list = kwargs.get('feature_list', [])
        
        # Si vide, tenter de récupérer depuis d'autres sources
        if not feature_list:
            df = kwargs['df']
            target_column = kwargs.get('target_column')
            
            if target_column:
                # Exclure la target
                feature_list = [col for col in df.columns if col != target_column]
            else:
                # Clustering: toutes les colonnes numériques
                feature_list = df.select_dtypes(include=['number']).columns.tolist()
            
            if not feature_list:
                raise ValueError("❌ Impossible de déterminer feature_list automatiquement")
            
            logger.warning(f"⚠️ feature_list vide, récupération auto: {len(feature_list)} features")
        
        context = MLTrainingContext(
            df=kwargs['df'],
            target_column=kwargs.get('target_column'),
            feature_list=feature_list,  # Utilisation de la feature_list déterminée
            task_type=kwargs.get('task_type', 'classification'),
            test_size=kwargs.get('test_size', 0.2),
            model_names=kwargs.get('model_names', []),
            optimize_hyperparams=kwargs.get('optimize', False),
            preprocessing_config=kwargs.get('preprocessing_choices'),
            use_smote=kwargs.get('use_smote', False),
            metadata=kwargs
        )
        
        result = ml_training_orchestrator.train(context)
        
        # Retour au format attendu par le code existant
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