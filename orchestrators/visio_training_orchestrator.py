"""
Orchestrateur d'entraînement Computer Vision
Gère la logique métier complète d'entraînement avec MLflow
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass

from src.models.computer_vision_training import (
    ComputerVisionTrainer,
    AnomalyAwareTrainer,
    ModelConfig,
    TrainingConfig,
    ModelType,
    OptimizerType,  
    SchedulerType  
)
from src.data.computer_vision_preprocessing import DataPreprocessor
from monitoring.mlflow_vision_tracker import cv_mlflow_tracker
from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingContext:
    """Contexte d'entraînement complet"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    model_config: Dict[str, Any]
    training_config: TrainingConfig
    preprocessing_config: Dict[str, Any]
    callbacks: list = None
    anomaly_type: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class TrainingResult:
    """Résultat d'entraînement standardisé"""
    success: bool
    model: Optional[Any] = None
    history: Optional[Dict] = None
    preprocessor: Optional[DataPreprocessor] = None
    mlflow_run_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class ComputerVisionTrainingOrchestrator:
    """
    Orchestrateur d'entraînement Computer Vision.
    
    Responsabilités:
    - Coordination du workflow d'entraînement complet
    - Intégration avec MLflow
    - Gestion du preprocessing
    - Gestion des erreurs et logging
    """
    
    def __init__(self):
        self.logger = logger
    
    def train(self, context: TrainingContext) -> TrainingResult:
        """
        Lance l'entraînement complet avec contexte.
        
        Args:
            context: TrainingContext avec toutes les données et configs
            
        Returns:
            TrainingResult avec modèle, historique et métadonnées
        """
        run_id = None
        
        try:
            
            # ========================================
            # 0. CONVERSION ROBUSTE DES CONFIGS (NOUVEAU)
            # ========================================
            context.training_config = self._ensure_training_config_object(context.training_config)

            # ========================================
            # 1. VALIDATION DES DONNÉES
            # ========================================
            self._validate_training_context(context)
            
            # ========================================
            # 2. DÉMARRAGE RUN MLFLOW
            # ========================================
            run_id = self._start_mlflow_run(context)
            
            # ========================================
            # 3. LOG CONFIGURATION
            # ========================================
            self._log_configuration_to_mlflow(context)
            
            # ========================================
            # 4. CRÉATION DU TRAINER
            # ========================================
            trainer = self._create_trainer(context)
            
            # ========================================
            # 5. ENTRAÎNEMENT
            # ========================================
            result = self._execute_training(trainer, context)
            
            # VÉRIFICATION SUCCÈS
            if not result.get('success', False):  # ✅ Accès dict au lieu de .success
                cv_mlflow_tracker.end_run("FAILED")
                return TrainingResult(
                    success=False,
                    error=result.get('error', 'Erreur inconnue'),  # ✅ Accès dict
                    mlflow_run_id=run_id
                )
            
            # ========================================
            # 6. RÉCUPÉRATION PREPROCESSOR
            # ========================================
            preprocessor = self._get_preprocessor(trainer, context)
            
            # ========================================
            # 7. LOG MÉTRIQUES ET ARTIFACTS
            # ========================================
            history = result['data']['history']  # Accès dict
            self._log_training_metrics(history)
            self._log_training_artifacts(trainer.model, preprocessor, context, run_id)
            
            # ========================================
            # 8. FINALISATION
            # ========================================
            cv_mlflow_tracker.end_run("FINISHED")
            
            # Construction résultat final
            final_history = self._build_final_history(history, context, run_id, preprocessor)
            
            return TrainingResult(
                success=True,
                model=trainer.model,
                history=final_history,
                preprocessor=preprocessor,
                mlflow_run_id=run_id,
                metadata={
                    "model_type": context.model_config["model_type"],
                    "total_epochs": history.get('total_epochs_trained', 0),
                    "best_epoch": history.get('best_epoch', 0)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Erreur orchestration entraînement: {e}", exc_info=True)
            
            if run_id:
                cv_mlflow_tracker.log_metrics({"training_failed": 1.0})
                cv_mlflow_tracker.end_run("FAILED")
            
            return TrainingResult(
                success=False,
                error=str(e),
                mlflow_run_id=run_id,
                metadata={
                    "error_type": type(e).__name__,
                    "X_train_shape": context.X_train.shape if context.X_train is not None else None,
                    "X_val_shape": context.X_val.shape if context.X_val is not None else None
                }
            )


    def _ensure_training_config_object(self, training_config: Any) -> TrainingConfig:
        """
        Convertit une configuration d'entraînement en objet TrainingConfig.
        - Validation stricte des EnumType (optimizer, scheduler)
        - Gestion des erreurs de conversion explicites
        - Fallback sécurisé sur valeurs par défaut
        
        Args:
            training_config: Configuration sous forme d'objet ou dict
            
        Returns:
            TrainingConfig: Objet de configuration typé
            
        Raises:
            ValueError: Si la configuration est invalide
        """
        try:
            # Cas 1: Déjà un objet TrainingConfig
            if isinstance(training_config, TrainingConfig):
                self.logger.info("✅ Configuration training: objet TrainingConfig détecté")
                return training_config
            
            # Cas 2: Dict venant de STATE
            elif isinstance(training_config, dict):
                self.logger.info("🔄 Conversion dict → TrainingConfig")
                
                # === VALIDATION ET CONVERSION OPTIMIZER ===
                optimizer_value = training_config.get('optimizer', 'adamw')
                
                # Si déjà OptimizerType, conserver
                if isinstance(optimizer_value, OptimizerType):
                    optimizer_enum = optimizer_value
                # Si string, valider et convertir
                elif isinstance(optimizer_value, str):
                    try:
                        optimizer_enum = OptimizerType(optimizer_value.lower())
                    except ValueError as e:
                        self.logger.error(
                            f"❌ Optimizer invalide: '{optimizer_value}'. "
                            f"Valeurs acceptées: {[e.value for e in OptimizerType]}"
                        )
                        raise ValueError(
                            f"Optimizer '{optimizer_value}' non supporté. "
                            f"Utilisez: {', '.join([e.value for e in OptimizerType])}"
                        ) from e
                else:
                    self.logger.warning(f"⚠️ Type optimizer inattendu: {type(optimizer_value)}, fallback adamw")
                    optimizer_enum = OptimizerType.ADAMW
                
                # === VALIDATION ET CONVERSION SCHEDULER ===
                scheduler_value = training_config.get('scheduler', 'reduce_on_plateau')
                
                if isinstance(scheduler_value, SchedulerType):
                    scheduler_enum = scheduler_value
                elif isinstance(scheduler_value, str):
                    try:
                        scheduler_enum = SchedulerType(scheduler_value.lower())
                    except ValueError as e:
                        self.logger.error(
                            f"❌ Scheduler invalide: '{scheduler_value}'. "
                            f"Valeurs acceptées: {[e.value for e in SchedulerType]}"
                        )
                        raise ValueError(
                            f"Scheduler '{scheduler_value}' non supporté. "
                            f"Utilisez: {', '.join([e.value for e in SchedulerType])}"
                        ) from e
                else:
                    self.logger.warning(f"⚠️ Type scheduler inattendu: {type(scheduler_value)}, fallback reduce_on_plateau")
                    scheduler_enum = SchedulerType.REDUCE_ON_PLATEAU
                
                # === CONSTRUCTION DICT AVEC VALEURS VALIDÉES ===
                config_dict = {
                    'epochs': int(training_config.get('epochs', 100)),
                    'batch_size': int(training_config.get('batch_size', 32)),
                    'learning_rate': float(training_config.get('learning_rate', 1e-4)),
                    'weight_decay': float(training_config.get('weight_decay', 0.01)),
                    'gradient_clip': float(training_config.get('gradient_clip', 1.0)),
                    'optimizer': optimizer_enum,  # ✅ Enum validé
                    'scheduler': scheduler_enum,  # ✅ Enum validé
                    'early_stopping_patience': int(training_config.get('early_stopping_patience', 15)),
                    'reduce_lr_patience': int(training_config.get('reduce_lr_patience', 8)),
                    'min_lr': float(training_config.get('min_lr', 1e-7)),
                    'use_class_weights': bool(training_config.get('use_class_weights', False)),
                    'use_mixed_precision': bool(training_config.get('use_mixed_precision', False)),
                    'deterministic': bool(training_config.get('deterministic', True)),
                    'seed': int(training_config.get('seed', 42)),
                    'num_workers': int(training_config.get('num_workers', 0)),
                    'pin_memory': bool(training_config.get('pin_memory', False)),
                    'checkpoint_dir': training_config.get('checkpoint_dir'),
                    'save_best_only': bool(training_config.get('save_best_only', True))
                }
                
                # === VALIDATION DES VALEURS CRITIQUES ===
                if config_dict['epochs'] <= 0:
                    raise ValueError(f"epochs doit être > 0, reçu: {config_dict['epochs']}")
                if config_dict['batch_size'] <= 0:
                    raise ValueError(f"batch_size doit être > 0, reçu: {config_dict['batch_size']}")
                if not (0 < config_dict['learning_rate'] < 1):
                    raise ValueError(f"learning_rate doit être dans ]0,1[, reçu: {config_dict['learning_rate']}")
                if config_dict['gradient_clip'] <= 0:
                    raise ValueError(f"gradient_clip doit être > 0, reçu: {config_dict['gradient_clip']}")
                
                # === CRÉATION DE L'OBJET ===
                training_config_obj = TrainingConfig(**config_dict)
                
                self.logger.info(
                    f"✅ Configuration training convertie avec succès - "
                    f"epochs: {training_config_obj.epochs}, "
                    f"batch_size: {training_config_obj.batch_size}, "
                    f"lr: {training_config_obj.learning_rate}, "
                    f"optimizer: {training_config_obj.optimizer.value}, "
                    f"scheduler: {training_config_obj.scheduler.value}"
                )
                
                return training_config_obj
            
            # Cas 3: Format inconnu
            else:
                raise ValueError(f"Format de configuration non supporté: {type(training_config)}")
                
        except ValueError as e:
            # Erreurs de validation remontées telles quelles
            self.logger.error(f"❌ Erreur validation TrainingConfig: {e}")
            raise
        
        except Exception as e:
            # Autres erreurs wrappées
            self.logger.error(f"❌ Erreur inattendue conversion TrainingConfig: {e}", exc_info=True)
            raise ValueError(f"Configuration d'entraînement invalide: {str(e)}")


    def _validate_training_context(self, context: TrainingContext):
        """Valide le contexte d'entraînement"""
        if context.X_train is None or context.X_val is None:
            raise ValueError("Données d'entraînement ou validation manquantes")
        
        if len(context.X_train) == 0 or len(context.X_val) == 0:
            raise ValueError("Datasets vides")
        
        self.logger.info(
            f"Données validées - X_train_shape: {context.X_train.shape}, X_val_shape: {context.X_val.shape}"
        )
    
    def _start_mlflow_run(self, context: TrainingContext) -> Optional[str]:
        """Démarre le run MLflow avec tags enrichis"""
        import time
        
        tags = {
            "anomaly_type": context.anomaly_type or "classification",
            "dataset_size": str(len(context.X_train)),
            "train_val_split": f"{len(context.X_train)}/{len(context.X_val)}",
            "model_type": context.model_config["model_type"]
        }
        
        # Ajouter métadonnées personnalisées
        if context.metadata:
            tags.update({
                k: str(v) for k, v in context.metadata.items()
                if isinstance(v, (str, int, float, bool))
            })
        
        run_name = f"{context.model_config['model_type']}_{int(time.time())}"
        return cv_mlflow_tracker.start_run(run_name=run_name, tags=tags)
    
    def _log_configuration_to_mlflow(self, context: TrainingContext):
        """Log toutes les configurations dans MLflow"""
        # Model config
        model_params = context.model_config.get("model_params", {})
        cv_mlflow_tracker.log_model_config({
            "model_type": context.model_config["model_type"],
            **{k: v for k, v in model_params.items() if isinstance(v, (str, int, float, bool))}
        })
        
        # Training config
        training_dict = {
            "epochs": context.training_config.epochs,
            "batch_size": context.training_config.batch_size,
            "learning_rate": context.training_config.learning_rate,
            "optimizer": context.training_config.optimizer.value if hasattr(context.training_config.optimizer, 'value') else str(context.training_config.optimizer),
            "weight_decay": getattr(context.training_config, 'weight_decay', 0),
            "early_stopping_patience": getattr(context.training_config, 'early_stopping_patience', None)
        }
        cv_mlflow_tracker.log_training_config(training_dict)
        
        # Preprocessing config
        if context.preprocessing_config:
            cv_mlflow_tracker.log_training_config({
                "preprocessing_strategy": context.preprocessing_config.get("strategy", "standardize"),
                "augmentation_enabled": context.preprocessing_config.get("augmentation_enabled", False)
            })
    
    def _create_trainer(self, context: TrainingContext):
        """
        Crée le trainer approprié selon le type de modèle.     
        Propagation de preprocessing_config au trainer    
        Args:
            context: Contexte d'entraînement contenant les configurations et données.    
        Returns:
            Instance de ComputerVisionTrainer ou AnomalyAwareTrainer.
        """
        try:
            # === ÉTAPE 0: CALCUL input_size DEPUIS le contexte ===
            input_size = self._compute_input_size_from_context(context)
            
            logger.info(f"✅ input_size calculée depuis le contexte: {input_size}")
            
            # === ÉTAPE 1: RÉCUPÉRATION MODE ===
            split_config = getattr(context, 'split_config', None)
            detected_mode = None
            
            if split_config and isinstance(split_config, dict):
                detected_mode = split_config.get('mode')
                logger.info(f"🔍 Mode depuis split_config: {detected_mode}")
            else:
                logger.warning("⚠️ split_config absent ou invalide")
            
            # === ÉTAPE 2: RÉCONCILIATION AVEC anomaly_type ===
            is_unsupervised = False
            final_anomaly_type = context.anomaly_type
            
            if context.anomaly_type:
                is_unsupervised = True
                logger.info(f"✅ Mode NON-SUPERVISÉ confirmé via anomaly_type: {context.anomaly_type}")
                
                if detected_mode and detected_mode != "unsupervised":
                    logger.warning(
                        f"⚠️ INCOHÉRENCE: anomaly_type={context.anomaly_type} "
                        f"mais split_config.mode={detected_mode}. "
                        f"Priorité donnée à anomaly_type."
                    )
            
            elif detected_mode == "unsupervised":
                is_unsupervised = True
                final_anomaly_type = "structural"
                logger.info(
                    f"✅ Mode NON-SUPERVISÉ détecté via split_config. "
                    f"Anomaly type par défaut: {final_anomaly_type}"
                )
            
            else:
                is_unsupervised = False
                logger.info("✅ Mode SUPERVISÉ confirmé (classification)")
                
                num_classes = context.model_config.get("model_params", {}).get("num_classes", 2)
                if num_classes < 2:
                    raise ValueError(
                        f"❌ Classification supervisée nécessite num_classes >= 2, "
                        f"reçu: {num_classes}"
                    )
            
            # === ÉTAPE 3: CRÉATION ModelConfig AVEC input_size ===
            model_params = context.model_config.get("model_params", {})
            
            model_config = ModelConfig(
                model_type=ModelType(context.model_config["model_type"]),
                num_classes=model_params.get("num_classes", 2),
                input_channels=model_params.get("input_channels", 3),
                dropout_rate=model_params.get("dropout_rate", 0.5),
                base_filters=model_params.get("base_filters", 32),
                latent_dim=model_params.get("latent_dim", 128),
                num_stages=model_params.get("num_stages", 4),
                input_size=input_size
            )
            
            callbacks = context.callbacks or []
            
            # === ÉTAPE 4: CRÉATION DU TRAINER ===
            if is_unsupervised:
                logger.info(f"🔍 Création AnomalyAwareTrainer: anomaly_type={final_anomaly_type}")
                
                trainer = AnomalyAwareTrainer(
                    anomaly_type=final_anomaly_type,
                    model_config=model_config,
                    training_config=context.training_config,
                    taxonomy_config=None,
                    callbacks=callbacks
                )
            else:
                logger.info(f"🎯 Création ComputerVisionTrainer: num_classes={model_config.num_classes}")
                
                trainer = ComputerVisionTrainer(
                    model_config=model_config,
                    training_config=context.training_config,
                    callbacks=callbacks
                )
            
            # Propagation preprocessing_config au trainer
            if hasattr(context, 'preprocessing_config') and context.preprocessing_config:
                trainer.preprocessing_config = context.preprocessing_config
                logger.info(
                    f"✅ preprocessing_config propagé au trainer: "
                    f"target_size={context.preprocessing_config.get('target_size', None)}"
                )
            else:
                logger.warning("⚠️ Aucun preprocessing_config dans le contexte")
            
            return trainer
        
        except Exception as e:
            logger.error(f"❌ Erreur création trainer: {e}", exc_info=True)
            raise ValueError(f"Impossible de créer le trainer: {str(e)}") from e
        
        
    def _compute_input_size_from_context(self, context: TrainingContext) -> Tuple[int, int]:
        """
        Calcule input_size avec VALIDATION de cohérence.   
        PRIORITÉ:
        1. preprocessing_config.target_size (si resize activé)
        2. X_train.shape (taille originale)   
        Validation que target_size est raisonnable vs données réelles
        """
        # 1. Si resize activé → utiliser target_size
        preprocessing_config = getattr(context, 'preprocessing_config', {})
        target_size = preprocessing_config.get('target_size', None)
        
        if target_size:
            # Vérifier que target_size est cohérent
            X_train_size = self._get_data_size(context.X_train)
            
            if X_train_size is not None:
                h_data, w_data = X_train_size
                h_target, w_target = target_size
                
                # Warning si ratio aspect change dramatiquement
                ratio_data = h_data / w_data
                ratio_target = h_target / w_target
                
                if abs(ratio_data - ratio_target) > 0.3:
                    logger.warning(
                        f"⚠️ ATTENTION: Changement important de ratio aspect - "
                        f"Data: {h_data}x{w_data} (ratio={ratio_data:.2f}), "
                        f"Target: {h_target}x{w_target} (ratio={ratio_target:.2f}). "
                        f"Cela peut causer des distorsions."
                    )
                
                # Warning si resize extrême (> 4x upscale ou < 0.25x downscale)
                scale_h = h_target / h_data
                scale_w = w_target / w_data
                
                if scale_h > 4 or scale_w > 4:
                    logger.warning(
                        f"⚠️ Upscale extrême détecté: {h_data}x{w_data} → {h_target}x{w_target} "
                        f"(scale_h={scale_h:.2f}x, scale_w={scale_w:.2f}x). "
                        f"Perte de qualité possible."
                    )
                
                if scale_h < 0.25 or scale_w < 0.25:
                    logger.warning(
                        f"⚠️ Downscale extrême détecté: {h_data}x{w_data} → {h_target}x{w_target} "
                        f"(scale_h={scale_h:.2f}x, scale_w={scale_w:.2f}x). "
                        f"Perte d'information possible."
                    )
            
            logger.info(f"✅ input_size depuis resize config (VALIDÉ): {target_size}")
            return target_size
        
        # 2. Sinon, calculer depuis X_train 
        logger.info("ℹ️ Pas de resize, calcul depuis données originales")
        return self._compute_input_size_from_data(context)
   

    def _get_data_size(self, X: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Extrait (height, width) depuis les données.
        Gère automatiquement channels_first et channels_last.    
        Returns:
            (height, width) ou None si impossible
        """
        try:
            if X is None or len(X) == 0:
                return None            
            shape = X.shape           
            if len(shape) != 4:
                return None
            
            # Format channels_last: (N, H, W, C) avec C petit (1, 3, 4)
            if shape[-1] in [1, 3, 4]:
                return (shape[1], shape[2])
            
            # Format channels_first: (N, C, H, W) avec C petit
            elif shape[1] in [1, 3, 4]:
                return (shape[2], shape[3])
            
            else:
                # Ambiguïté: assumer channels_last par défaut
                logger.warning(f"⚠️ Format ambigu {shape}, assume channels_last")
                return (shape[1], shape[2])
        
        except Exception as e:
            logger.debug(f"Impossible d'extraire size: {e}")
            return None

    def _compute_input_size_from_data(self, context: TrainingContext) -> Tuple[int, int]:
        """
        Calcule input_size depuis les données réelles. 
        """
        try:
            X_train = context.X_train
            
            if X_train is None or len(X_train) == 0:
                raise ValueError("X_train est None ou vide")
            
            # Détection du format: (N, H, W, C) ou (N, C, H, W)
            shape = X_train.shape
            
            if len(shape) != 4:
                raise ValueError(f"X_train doit être 4D, reçu: {shape}")
            
            # Format channels_last: (N, H, W, C) avec C petit (1, 3, 4)
            if shape[-1] in [1, 3, 4]:
                height, width = shape[1], shape[2]
                logger.info(f"Format détecté: channels_last (N,H,W,C)")
            
            # Format channels_first: (N, C, H, W) avec C petit
            elif shape[1] in [1, 3, 4]:
                height, width = shape[2], shape[3]
                logger.info(f"Format détecté: channels_first (N,C,H,W)")
            
            else:
                # Ambiguïté: assumer channels_last par défaut
                height, width = shape[1], shape[2]
                logger.warning(
                    f"⚠️ Format ambigu {shape}, assume channels_last. "
                    f"Si incorrect, spécifiez input_size explicitement."
                )
            
            # Validation
            if height < 32 or width < 32:
                raise ValueError(
                    f"Dimensions trop petites: ({height}, {width}). Minimum: (32, 32)"
                )
            
            if height > 1024 or width > 1024:
                logger.warning(
                    f"⚠️ Dimensions très grandes: ({height}, {width}). "
                    f"Considérez de redimensionner pour économiser mémoire."
                )
            
            logger.info(f"✅ input_size calculée: ({height}, {width}) depuis X_train.shape={shape}")
            
            return (height, width)
        
        except Exception as e:
            logger.error(f"❌ Erreur calcul input_size: {e}", exc_info=True)
            raise ValueError(
                f"Impossible de calculer input_size depuis les données: {str(e)}"
            ) from e
    
    def _execute_training(self, trainer, context: TrainingContext):
        """
        Exécute l'entraînement et normalise le format de retour.
        Retour TOUJOURS au format dict standardisé
        """
        try:
            # Exécution selon le type de trainer
            if context.anomaly_type:
                raw_result = trainer.train(
                    context.X_train,
                    context.y_train,
                    context.X_val,
                    context.y_val
                )
            else:
                raw_result = trainer.fit(
                    context.X_train,
                    context.y_train,
                    context.X_val,
                    context.y_val
                )
            
            # ========================================================================
            # NORMALISATION ROBUSTE - Gère TOUS les cas possibles
            # ========================================================================
            
            # CAS 1 : Déjà un dict
            if isinstance(raw_result, dict):
                result = raw_result
                self.logger.info(f"✅ Résultat format dict (legacy)")
            
            # CAS 2 : Objet Result avec attribut .data
            elif hasattr(raw_result, 'data'):
                self.logger.info(f"✅ Résultat format Result, extraction .data")
                
                # Extraction du dict depuis Result
                if isinstance(raw_result.data, dict):
                    result = {
                        'success': getattr(raw_result, 'success', True),
                        'data': raw_result.data,
                        'error': getattr(raw_result, 'error', None),
                        'metadata': getattr(raw_result, 'metadata', {})
                    }
                else:
                    # Cas où .data n'est pas un dict (objet complexe)
                    result = {
                        'success': getattr(raw_result, 'success', True),
                        'data': {
                            'history': getattr(raw_result.data, 'history', {}) if hasattr(raw_result.data, 'history') else {}
                        },
                        'error': getattr(raw_result, 'error', None),
                        'metadata': getattr(raw_result, 'metadata', {})
                    }
            
            # CAS 3 : Objet avec success/data mais pas de .data direct
            elif hasattr(raw_result, 'success'):
                self.logger.info(f"✅ Résultat format objet custom")
                result = {
                    'success': raw_result.success,
                    'data': getattr(raw_result, 'history', {}) if hasattr(raw_result, 'history') else {},
                    'error': getattr(raw_result, 'error', None),
                    'metadata': getattr(raw_result, 'metadata', {})
                }
            
            # CAS 4 : Format inconnu
            else:
                self.logger.error(f"❌ Format résultat invalide: {type(raw_result)}")
                return {
                    'success': False,
                    'error': f"Format résultat invalide: {type(raw_result)}",
                    'data': {},
                    'metadata': {}
                }
            
            # ========================================================================
            # VALIDATION ET NORMALISATION DU DICT
            # ========================================================================
            
            # S'assurer que les clés essentielles existent
            if 'success' not in result:
                result['success'] = 'error' not in result or result['error'] is None
            
            if 'data' not in result:
                result['data'] = {}
            
            if 'error' not in result:
                result['error'] = None
            
            if 'metadata' not in result:
                result['metadata'] = {}
            
            # S'assurer que data est un dict
            if not isinstance(result['data'], dict):
                result['data'] = {'raw': result['data']}
            
            # S'assurer que data.history existe
            if 'history' not in result['data']:
                # Tenter de récupérer depuis trainer
                if hasattr(trainer, 'history') and trainer.history:
                    result['data']['history'] = trainer.history
                    self.logger.warning("⚠️ history récupérée depuis trainer.history")
                else:
                    # Créer un historique minimal
                    result['data']['history'] = {
                        'train_loss': [],
                        'val_loss': [],
                        'total_epochs_trained': 0,
                        'best_epoch': 0,
                        'training_time': 0.0
                    }
                    self.logger.warning("⚠️ Historique vide créé (fallback)")
            
            # ========================================================================
            # LOG DÉTAILLÉ DU RÉSULTAT FINAL
            # ========================================================================
            self.logger.info(
                f"✅ Entraînement terminé - Format normalisé - "
                f"success: {result['success']}, "
                f"has_data: {bool(result.get('data'))}, "
                f"has_history: {'history' in result.get('data', {})}, "
                f"has_error: {result.get('error') is not None}, "
                f"result_keys: {list(result.keys())}, "
                f"data_keys: {list(result.get('data', {}).keys())}"
            )
            
            # ========================================================================
            # VALIDATION FINALE
            # ========================================================================
            if result['success'] and not result.get('data', {}).get('history'):
                self.logger.error("❌ Succès déclaré mais historique manquant!")
                result['success'] = False
                result['error'] = "Historique d'entraînement manquant"
            
            return result  
            
        except Exception as e:
            self.logger.error(f"❌ Erreur exécution entraînement: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'data': {},
                'metadata': {
                    'exception_type': type(e).__name__,
                    'X_train_shape': context.X_train.shape if context.X_train is not None else None,
                    'X_val_shape': context.X_val.shape if context.X_val is not None else None
                }
            }
    
    def _get_preprocessor(self, trainer, context: TrainingContext) -> DataPreprocessor:
        """Récupère ou crée le preprocessor"""
        preprocessor = getattr(trainer, 'preprocessor', None)
        
        if preprocessor is None:
            self.logger.warning("Aucun preprocessor trouvé, création fallback")
            preprocessor = DataPreprocessor(
                strategy=context.preprocessing_config.get("strategy", "standardize"),
                auto_detect_format=True
            )
            preprocessor.fit(context.X_train)
        
        return preprocessor
    
    def _log_training_metrics(self, history: Dict):
        """Log les métriques d'entraînement dans MLflow"""
        # Métriques par epoch
        for epoch in range(len(history.get('train_loss', []))):
            metrics = {}
            
            for key in ['train_loss', 'val_loss', 'val_accuracy', 'val_f1', 'learning_rates']:
                if key in history and epoch < len(history[key]):
                    metric_name = 'learning_rate' if key == 'learning_rates' else key
                    metrics[metric_name] = history[key][epoch]
            
            if metrics:
                cv_mlflow_tracker.log_metrics(metrics, step=epoch)
        
        # Métriques finales
        final_metrics = {
            "best_val_loss": float(history.get('best_val_loss', 0)),
            "training_time_seconds": float(history.get('training_time', 0)),
            "total_epochs": int(history.get('total_epochs_trained', 0)),
            "best_epoch": int(history.get('best_epoch', 0))
        }
        
        if history.get('val_accuracy'):
            final_metrics['final_val_accuracy'] = float(history['val_accuracy'][-1])
            final_metrics['best_val_accuracy'] = float(max(history['val_accuracy']))
        
        if history.get('val_f1'):
            final_metrics['final_val_f1'] = float(history['val_f1'][-1])
            final_metrics['best_val_f1'] = float(max(history['val_f1']))
        
        if history.get('early_stopping_triggered'):
            final_metrics['early_stopping'] = 1.0
        
        cv_mlflow_tracker.log_metrics(final_metrics)
        
        # Log courbes
        cv_mlflow_tracker.log_training_curves(history)
    
    def _log_training_artifacts(
        self,
        model,
        preprocessor: DataPreprocessor,
        context: TrainingContext,
        run_id: str
    ):
        """
        Log modèle et artifacts dans MLflow avec gestion robuste des erreurs. 
        - Try-except sur chaque artifact individuellement
        - Vérification type preprocessor avant pickle
        - Fallback si pickle échoue
        - Logs détaillés des échecs
        """
        import time
        import pickle
        
        additional_files = {}
        
        # Preprocessor avec gestion erreur pickle
        if preprocessor is not None:
            try:
                # Vérifier que le preprocessor est picklable
                test_pickle = pickle.dumps(preprocessor)
                pickle.loads(test_pickle)  # Vérification round-trip
                
                additional_files['preprocessor.pkl'] = preprocessor
                self.logger.info("✅ Preprocessor ajouté aux artifacts")
            except (pickle.PicklingError, TypeError, AttributeError) as e:
                self.logger.warning(f"⚠️ Preprocessor non-picklable: {e}")
                # Sauvegarder uniquement la config
                try:
                    config = preprocessor.get_config()
                    additional_files['preprocessor_config.json'] = config
                    self.logger.info("✅ Config preprocessor sauvegardée (objet non-picklable)")
                except Exception as e2:
                    self.logger.error(f"❌ Impossible de sauvegarder config preprocessor: {e2}")
            except Exception as e:
                self.logger.error(f"❌ Erreur inattendue preprocessor: {e}")
        else:
            self.logger.warning("⚠️ Aucun preprocessor à sauvegarder")
        
        # Config preprocessor safe même si preprocessor None
        if preprocessor is not None and 'preprocessor_config.json' not in additional_files:
            try:
                additional_files['preprocessor_config.json'] = preprocessor.get_config()
            except Exception as e:
                self.logger.warning(f"⚠️ Impossible d'extraire config preprocessor: {e}")
                additional_files['preprocessor_config.json'] = {"error": "config_unavailable"}
        
        # Model config avec conversion safe
        try:
            model_config_dict = {
                "model_type": str(context.model_config.get("model_type", "unknown")),
                "model_params": {},
                "anomaly_type": str(context.anomaly_type) if context.anomaly_type else None,
                "training_config": {}
            }
            
            # Extraction safe model_params
            model_params = context.model_config.get("model_params", {})
            for k, v in model_params.items():
                try:
                    # Convertir en types sérialisables
                    if isinstance(v, (int, float, str, bool, type(None))):
                        model_config_dict["model_params"][k] = v
                    elif isinstance(v, (list, tuple)):
                        model_config_dict["model_params"][k] = list(v)
                    elif isinstance(v, dict):
                        model_config_dict["model_params"][k] = dict(v)
                    else:
                        model_config_dict["model_params"][k] = str(v)
                except Exception:
                    model_config_dict["model_params"][k] = "non_serializable"
            
            # Extraction safe training_config
            try:
                model_config_dict["training_config"] = {
                    "epochs": int(context.training_config.epochs),
                    "batch_size": int(context.training_config.batch_size),
                    "learning_rate": float(context.training_config.learning_rate),
                    "optimizer": str(context.training_config.optimizer.value if hasattr(context.training_config.optimizer, 'value') else context.training_config.optimizer),
                    "early_stopping_patience": getattr(context.training_config, 'early_stopping_patience', None)
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur extraction training_config: {e}")
                model_config_dict["training_config"] = {"error": "extraction_failed"}
            
            additional_files['model_config.json'] = model_config_dict
            self.logger.info("✅ Model config ajoutée aux artifacts")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur création model_config: {e}")
            additional_files['model_config.json'] = {"error": str(e)}
        
        # Log artifacts avec gestion erreur globale
        try:
            model_filename = f"model_{run_id or 'local'}_{int(time.time())}.pt"
            
            cv_mlflow_tracker.log_model_artifact(
                model=model,
                filename=model_filename,
                additional_files=additional_files
            )
            
            self.logger.info(f"✅ Artifacts loggés: {model_filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Échec log artifacts MLflow: {e}", exc_info=True)
            # Ne pas faire échouer tout l'entraînement si MLflow fail
            self.logger.warning("⚠️ Entraînement continue malgré échec log MLflow")

    
    def _build_final_history(
        self,
        history_data: Dict,
        context: TrainingContext,
        run_id: str,
        preprocessor: DataPreprocessor
    ) -> Dict:
        """
        Construit l'historique final normalisé.    
        Gestion robuste des types None et validation
        """
        
        def safe_float(value, default=0.0):
            """Conversion safe en float avec gestion None"""
            try:
                if value is None:
                    return default
                if isinstance(value, (list, tuple, np.ndarray)):
                    if len(value) == 0:
                        return default
                    value = value[-1]
                f = float(value)
                if np.isnan(f) or np.isinf(f):
                    return default
                return f
            except (ValueError, TypeError):
                return default
        
        def safe_int(value, default=0):
            """Conversion safe en int avec gestion None"""
            try:
                if value is None:
                    return default
                return int(value)
            except (ValueError, TypeError):
                return default
        
        def safe_list_float(values, default_list=None):
            """Conversion safe liste de floats avec gestion None"""
            if default_list is None:
                default_list = []
            
            try:
                if values is None or (hasattr(values, '__len__') and len(values) == 0):
                    return default_list
                
                result = []
                for v in values:
                    try:
                        f = float(v)
                        if not (np.isnan(f) or np.isinf(f)):
                            result.append(f)
                        else:
                            result.append(0.0)
                    except (ValueError, TypeError):
                        result.append(0.0)
                
                return result if result else default_list
            except Exception:
                return default_list
        
        try:
            # ========================================================================
            # EXTRACTION SAFE DES MÉTRIQUES
            # ========================================================================
            
            # Listes de métriques avec fallbacks
            train_loss = safe_list_float(history_data.get('train_loss'), [0.0])
            val_loss = safe_list_float(history_data.get('val_loss'), [0.0])
            val_accuracy = safe_list_float(history_data.get('val_accuracy'), [])
            val_f1 = safe_list_float(history_data.get('val_f1'), [])
            learning_rates = safe_list_float(history_data.get('learning_rates'), [])
            
            # Valeurs scalaires avec fallbacks
            best_epoch = safe_int(history_data.get('best_epoch'), 0)
            best_val_loss = safe_float(history_data.get('best_val_loss'), float('inf'))
            training_time = safe_float(history_data.get('training_time'), 0.0)
            total_epochs = safe_int(history_data.get('total_epochs_trained'), len(train_loss))
            early_stopping = bool(history_data.get('early_stopping_triggered', False))
            
            # Model type safe
            model_type_value = history_data.get('model_type') or context.model_config.get("model_type")
            model_type_str = str(model_type_value) if model_type_value else "unknown"
            
            # Input shape safe
            input_shape = history_data.get('input_shape')
            if input_shape is None and context.X_train is not None:
                input_shape = context.X_train.shape[1:]
            
            # ========================================================================
            # PREPROCESSOR CONFIG SAFE
            # ========================================================================
            preprocessor_config = None
            preprocessor_available = False
            
            if preprocessor is not None:
                preprocessor_available = True
                try:
                    preprocessor_config = preprocessor.get_config()
                except Exception as e:
                    self.logger.warning(f"⚠️ Impossible d'extraire config preprocessor: {e}")
                    preprocessor_config = {"error": "config_unavailable"}
            
            # ========================================================================
            # CONSTRUCTION DU DICT FINAL
            # ========================================================================
            final_history = {
                'success': True,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'learning_rates': learning_rates,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'training_time': training_time,
                'total_epochs_trained': total_epochs,
                'early_stopping_triggered': early_stopping,
                'model_type': model_type_str,
                'input_shape': input_shape,
                'anomaly_type': str(context.anomaly_type) if context.anomaly_type else None,
                'preprocessor_available': preprocessor_available,
                'preprocessor_config': preprocessor_config,
                'mlflow_run_id': str(run_id) if run_id else None
            }
            
            self.logger.info(
                f"✅ Historique construit: {total_epochs} epochs, "
                f"best_loss: {best_val_loss:.4f}, preprocessor: {preprocessor_available}"
            )
            
            return final_history
            
        except Exception as e:
            self.logger.error(f"❌ Erreur construction historique: {e}", exc_info=True)
            
            # FALLBACK : Historique minimal valide
            return {
                'success': True,
                'train_loss': [0.0],
                'val_loss': [0.0],
                'val_accuracy': [],
                'val_f1': [],
                'learning_rates': [],
                'best_epoch': 0,
                'best_val_loss': float('inf'),
                'training_time': 0.0,
                'total_epochs_trained': 0,
                'early_stopping_triggered': False,
                'model_type': 'unknown',
                'input_shape': None,
                'anomaly_type': None,
                'preprocessor_available': False,
                'preprocessor_config': None,
                'mlflow_run_id': None,
                'error': str(e)
            }

# Instance globale
training_orchestrator = ComputerVisionTrainingOrchestrator()