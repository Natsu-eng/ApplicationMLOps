"""
Production-Ready Computer Vision Training Pipeline
Version 1.0 - Enterprise Grade
Architecture:
- Modulaire et extensible
"""
import time
import warnings
import copy
from dataclasses import dataclass, field
from typing import Counter, Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from pathlib import Path

import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight # type: ignore

from src.config.model_config import ModelConfig, ModelType
from src.config.training_config import TrainingConfig, OptimizerType, SchedulerType

from src.models.trainer import ComputerVisionTrainer, OptimizerFactory, SchedulerFactory

from src.data.computer_vision_preprocessing import DataLoaderFactory, DataPreprocessor, DataValidator, Result
from src.models.computer_vision.model_builder import ModelBuilder
from src.shared.logging import get_logger
from utils.callbacks import LoggingCallback, TrainingCallback
from utils.device_manager import DeviceManager

warnings.filterwarnings('ignore', category=UserWarning)

logger = get_logger(__name__)


# =================================
# INTÉGRATION AVEC ANOMALY TAXONOMY  
# =================================

class AnomalyAwareTrainer:
    """
    Trainer spécialisé pour la détection d'anomalies avec taxonomie.
    
    NOUVEAU: Intégration complète restaurée depuis l'ancienne version
    """
    
    def __init__(
        self,
        anomaly_type: Optional[str] = None,  
        *,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        taxonomy_config: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        auto_detect_from_state: bool = True
    ):
        """
        Initialise le trainer pour anomalies.
        
        Args:
            anomaly_type: Type d'anomalie (structural, visual, geometric) ou None pour auto
            model_config: Configuration du modèle (optionnel, sinon auto-configuré)
            training_config: Configuration d'entraînement (optionnel, sinon auto-configuré)
            taxonomy_config: Configuration de taxonomie personnalisée (optionnel)
            callbacks: Callbacks pour monitoring (optionnel)
            auto_detect_from_state: Active détection automatique depuis STATE si anomaly_type=None
        """
        from monitoring.state_managers import STATE
        
        # Détection automatique si nécessaire
        if anomaly_type is None and auto_detect_from_state:
            logger.info("🔍 Détection automatique du type d'anomalie depuis STATE")
            anomaly_type = self._detect_anomaly_type_from_state(STATE)
            
            if anomaly_type is None:
                logger.warning("⚠️ Impossible de détecter anomaly_type, fallback 'structural'")
                anomaly_type = "structural"
        
        elif anomaly_type is None:
            logger.info("ℹ️ Anomaly_type=None sans auto-détection, fallback 'structural'")
            anomaly_type = "structural"
        
        self.anomaly_type = anomaly_type
        self.taxonomy_config = taxonomy_config or self._get_default_taxonomy()
        self.callbacks = callbacks or []
        
        # Validation model_config si fourni
        if model_config is not None:
            if not isinstance(model_config.model_type, ModelType):
                raise ValueError(
                    f"model_config.model_type doit être une instance de ModelType, "
                    f"reçu: {type(model_config.model_type)}"
                )
            
            # Vérifier que c'est un modèle compatible anomalies
            valid_anomaly_models = [
                ModelType.CONV_AUTOENCODER,
                ModelType.VAE,
                ModelType.DENOISING_AE,
                ModelType.PATCH_CORE
            ]
            
            if model_config.model_type not in valid_anomaly_models:
                logger.warning(
                    f"⚠️ Modèle {model_config.model_type.value} inhabituel pour anomalies. "
                    f"Modèles recommandés: {[m.value for m in valid_anomaly_models]}"
                )
        
        # Configuration automatique ou manuelle
        if model_config is None or training_config is None:
            auto_model_config, auto_training_config = self._configure_for_anomaly()
            self.model_config = model_config or auto_model_config
            self.training_config = training_config or auto_training_config
            logger.info(f"🔧 Configuration automatique pour anomalie '{anomaly_type}'")
        else:
            self.model_config = model_config
            self.training_config = training_config
            logger.info(f"⚙️ Configuration manuelle pour anomalie '{anomaly_type}'")
        
        # Attributs pour compatibilité
        self.model: Optional[nn.Module] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.history: Dict[str, Any] = {}
    
    def _detect_anomaly_type_from_state(self, STATE) -> Optional[str]:
        """
        Détecte le type d'anomalie depuis STATE.data.
        """
        try:
            # Stratégie 1: Metadata explicite
            if hasattr(STATE.data, 'metadata') and STATE.data.metadata:
                anomaly_type = STATE.data.metadata.get('anomaly_type')
                if anomaly_type:
                    logger.info(f"✅ Anomaly type depuis metadata: {anomaly_type}")
                    return anomaly_type
            
            # Stratégie 2: Nom du dataset
            if hasattr(STATE.data, 'name') and STATE.data.name:
                name_lower = STATE.data.name.lower()
                
                if any(kw in name_lower for kw in ['crack', 'corrosion', 'deformation']):
                    logger.info(f"✅ Anomaly type depuis nom: structural")
                    return "structural"
                
                if any(kw in name_lower for kw in ['scratch', 'stain', 'color']):
                    logger.info(f"✅ Anomaly type depuis nom: visual")
                    return "visual"
                
                if any(kw in name_lower for kw in ['dimension', 'alignment', 'size']):
                    logger.info(f"✅ Anomaly type depuis nom: geometric")
                    return "geometric"
            
            # Stratégie 3: Structure MVTec AD
            if hasattr(STATE.data, 'structure') and STATE.data.structure:
                if STATE.data.structure.get('type') == 'mvtec_ad':
                    logger.info(f"✅ Structure MVTec AD détectée → structural")
                    return "structural"
            
            logger.warning("⚠️ Impossible de détecter anomaly_type automatiquement")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur détection anomaly_type: {e}", exc_info=True)
            return None
    
    def _get_default_taxonomy(self) -> Dict[str, Any]:
        """Taxonomie par défaut production-ready."""
        return {
            "structural": {
                "recommended_model": ModelType.CONV_AUTOENCODER,
                "difficulty": "high",
                "threshold": 0.90,
                "description": "Défauts structurels (cracks, corrosion, deformation)",
                "params": {
                    "latent_dim": 256,
                    "learning_rate": 1e-4,
                    "base_filters": 32,
                    "num_stages": 4
                },
                "training_params": {
                    "epochs": 100,
                    "batch_size": 32,
                    "early_stopping_patience": 15,
                    "use_class_weights": False  
                }
            },
            "visual": {
                "recommended_model": ModelType.DENOISING_AE,
                "difficulty": "medium",
                "threshold": 0.85,
                "description": "Défauts visuels (scratch, stain, discoloration)",
                "params": {
                    "latent_dim": 128,
                    "learning_rate": 1e-3,
                    "base_filters": 64,
                    "noise_factor": 0.1
                },
                "training_params": {
                    "epochs": 80,
                    "batch_size": 32,
                    "early_stopping_patience": 12,
                    "use_class_weights": False
                }
            },
            "geometric": {
                "recommended_model": ModelType.VAE,
                "difficulty": "low",
                "threshold": 0.95,
                "description": "Défauts géométriques (misalignment, dimension errors)",
                "params": {
                    "latent_dim": 64,
                    "learning_rate": 1e-3,
                    "base_filters": 32,
                    "beta": 1.0
                },
                "training_params": {
                    "epochs": 60,
                    "batch_size": 32,
                    "early_stopping_patience": 10,
                    "use_class_weights": False
                }
            }
        }
    
    def _configure_for_anomaly(self) -> Tuple[ModelConfig, TrainingConfig]:
        """Configure modèle et training selon le type d'anomalie."""
        category = self._get_anomaly_category(self.anomaly_type)
        config = self.taxonomy_config.get(category, self.taxonomy_config["structural"])
        
        logger.info(
            f"🔧 Configuration pour anomalie: {self.anomaly_type} (catégorie: {category}) - "
            f"difficulty: {config.get('difficulty')}, "
            f"recommended_model: {config['recommended_model'].value}"
        )
        
        # Configuration du modèle
        model_params = config.get("params", {})
        model_config = ModelConfig(
            model_type=config["recommended_model"],
            num_classes=2,
            input_channels=model_params.get("input_channels", 3),
            dropout_rate=model_params.get("dropout_rate", 0.0),
            base_filters=model_params.get("base_filters", 32),
            latent_dim=model_params.get("latent_dim", 256),
            num_stages=model_params.get("num_stages", 4)
        )
        
        # Configuration de l'entraînement depuis taxonomie
        training_params = config.get("training_params", {})
        training_config = TrainingConfig(
            epochs=training_params.get("epochs", 100),
            batch_size=training_params.get("batch_size", 32),
            learning_rate=model_params.get("learning_rate", 1e-4),
            optimizer=OptimizerType.ADAMW,
            scheduler=SchedulerType.REDUCE_ON_PLATEAU,
            early_stopping_patience=training_params.get("early_stopping_patience", 15),
            reduce_lr_patience=8,
            use_class_weights=False,
            gradient_clip=1.0,
            deterministic=True,
            seed=42,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(
            f"✅ Configuration générée - "
            f"latent_dim: {model_config.latent_dim}, "
            f"epochs: {training_config.epochs}, "
            f"lr: {training_config.learning_rate}"
        )
        
        return model_config, training_config
    
    def _get_anomaly_category(self, anomaly_type: str) -> str:
        """Détermine la catégorie d'anomalie avec mapping enrichi."""
        category_mappings = {
            "structural": [
                "crack", "corrosion", "deformation", "structural",
                "break", "fracture", "damage"
            ],
            "visual": [
                "scratch", "stain", "discoloration", "visual",
                "contamination", "dirt", "mark", "spot"
            ],
            "geometric": [
                "misalignment", "dimension_error", "geometric",
                "size", "position", "orientation"
            ]
        }
        
        anomaly_type_lower = anomaly_type.lower()
        
        for category, keywords in category_mappings.items():
            if anomaly_type_lower in keywords:
                logger.info(f"✅ Anomaly '{anomaly_type}' mappée à catégorie '{category}'")
                return category
        
        logger.warning(
            f"⚠️ Anomaly type '{anomaly_type}' non reconnue. "
            f"Fallback catégorie 'structural'"
        )
        return "structural"
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        callbacks: Optional[List[TrainingCallback]] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> Result:
        """
        Entraîne le modèle pour le type d'anomalie spécifique.
        
        Args:
            X_train, y_train: Données d'entraînement (normales + anomalies éventuelles)
            X_val, y_val: Données de validation
            callbacks: Callbacks additionnels à ajouter aux callbacks internes
            preprocessing_config: Configuration de préprocessing (notamment augmentation et target_size)
                                → Critique pour que l'augmentation fonctionne correctement
        
        Returns:
            Result avec succès ou erreur détaillée
        """
        try:
            logger.info(
                f"🚀 Début entraînement anomalies - "
                f"anomaly_type: {self.anomaly_type}, "
                f"model_type: {self.model_config.model_type.value}, "
                f"X_train_shape: {X_train.shape}, "
                f"X_val_shape: {X_val.shape}, "
                f"augmentation_enabled: {preprocessing_config.get('augmentation_enabled', False) if preprocessing_config else False}"
            )
            
            # Détection PatchCore → workflow spécial
            is_patchcore = self.model_config.model_type == ModelType.PATCH_CORE
            
            if is_patchcore:
                logger.info("🔍 PatchCore détecté - Utilisation du workflow natif fit() dédié")
                return self._train_patchcore(X_train, y_train, X_val, y_val, callbacks)
            
            # ====================================================================
            # Workflow standard : Autoencoders (ConvAE, VAE, DenoisingAE, etc.)
            # ====================================================================
            
            # Fusion des callbacks : ceux passés en paramètre + ceux définis dans la classe
            active_callbacks = (callbacks or []) + self.callbacks
            
            logger.info(
                f"🛠 Création ComputerVisionTrainer avec {len(active_callbacks)} callback(s) actif(s)"
            )
            
            # Instanciation du trainer générique
            trainer = ComputerVisionTrainer(
                model_config=self.model_config,
                training_config=self.training_config,
                callbacks=active_callbacks
            )
            
            # ====================================================================
            # APPEL CRITIQUE : Propagation du preprocessing_config
            # → Indispensable pour que target_size et augmentation soient pris en compte
            # ====================================================================
            result = trainer.fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                preprocessing_config=preprocessing_config  # ← CLÉ DE VOÛTE
            )
            
            # ====================================================================
            # Gestion du résultat
            # ====================================================================
            if result.success:
                # Récupération des artefacts entraînés
                result_data = result.data
                self.model = result_data['model']
                self.preprocessor = result_data['preprocessor']
                self.history = result_data['history']
                
                # Métadonnées utiles
                best_epoch = self.history.get('best_epoch', 0)
                best_val_loss = self.history.get('best_val_loss', float('inf'))
                total_time = result.metadata.get('training_time', 0)
                
                logger.info(
                    f"✅ Entraînement anomalies terminé avec succès !\n"
                    f" • Anomaly type      : {self.anomaly_type}\n"
                    f" • Modèle            : {self.model_config.model_type.value}\n"
                    f" • Best epoch        : {best_epoch}\n"
                    f" • Best val loss     : {best_val_loss:.6f}\n"
                    f" • Durée totale      : {total_time:.1f}s\n"
                    f" • Augmentation      : {preprocessing_config.get('augmentation_enabled', False) if preprocessing_config else False}\n"
                    f" • Target size       : {preprocessing_config.get('target_size') if preprocessing_config else 'auto'}"
                )
                
                return result
                
            else:
                # Échec de l'entraînement → propagation claire de l'erreur
                logger.error(f"❌ Entraînement anomalies échoué : {result.error}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Erreur critique dans train() anomalies : {e}", exc_info=True)
            return Result.err(f"Entraînement anomalies échoué : {str(e)}")


    def _train_patchcore(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        callbacks: Optional[List[TrainingCallback]] = None
    ) -> Result:
        """
        Entraînement spécifique pour PatchCore.   
        PatchCore n'a pas d'entraînement par epochs, mais construit une memory bank.
        """
        try:
            import time
            from src.models.computer_vision.model_builder import ModelBuilder
            from src.data.computer_vision_preprocessing import DataPreprocessor, DataLoaderFactory
            from utils.device_manager import DeviceManager
            
            start_time = time.time()
            
            # 1. Preprocessing
            logger.info("📊 Preprocessing données pour PatchCore")
            
            self.preprocessor = DataPreprocessor(
                strategy="standardize",
                auto_detect_format=True,
                target_size=None  # PatchCore accepte toutes les tailles
            )
            
            X_train_norm = self.preprocessor.fit_transform(X_train, output_format="channels_first")
            X_val_norm = self.preprocessor.transform(X_val, output_format="channels_first")
            
            # 2. Construction du modèle
            logger.info("Construction modèle PatchCore")
            
            builder = ModelBuilder(DeviceManager())
            result = builder.build(self.model_config)
            
            if not result.success:
                return result
            
            self.model = result.data
            
            # 3. Fit PatchCore (construction memory bank)
            logger.info("🔨 Construction memory bank PatchCore")
            
            train_loader = DataLoaderFactory.create(
                X_train_norm, y_train,
                batch_size=32,
                shuffle=False,  # Pas de shuffle pour PatchCore
                num_workers=0,
                pin_memory=False
            )
            
            # PatchCore.fit() construit la memory bank
            self.model.fit(train_loader)
            
            # 4. Évaluation sur validation
            logger.info("📊 Évaluation sur validation set")
            
            val_loader = DataLoaderFactory.create(
                X_val_norm, y_val,
                batch_size=32,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            # PatchCore.predict() retourne les scores d'anomalie
            val_scores = self.model.predict(val_loader)
            
            # Calcul métriques de validation
            threshold = np.percentile(val_scores, 95)
            val_preds = (val_scores > threshold).astype(int)
            
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            
            val_accuracy = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds, zero_division=0)
            
            try:
                val_auc = roc_auc_score(y_val, val_scores)
            except:
                val_auc = 0.0
            
            training_time = time.time() - start_time
            
            # 5. Construction de l'historique
            self.history = {
                'success': True,
                'model_type': 'patch_core',
                'is_autoencoder': False,
                'train_loss': [0.0],  # PatchCore n'a pas de loss d'entraînement
                'val_loss': [float(threshold)],  # Utiliser le threshold comme proxy
                'val_accuracy': [float(val_accuracy)],
                'val_f1': [float(val_f1)],
                'val_auc': [float(val_auc)],
                'learning_rates': [],
                'best_epoch': 0,
                'best_val_loss': float(threshold),
                'final_train_loss': 0.0,
                'training_time': training_time,
                'total_epochs_trained': 1,  # PatchCore = 1 "epoch" (fit unique)
                'early_stopping_triggered': False,
                'input_shape': tuple(self.preprocessor.original_shape_[1:]),
                'output_format': 'channels_first',
                'threshold': float(threshold),
                'training_config': {
                    'learning_rate': 0.0,  # PatchCore n'a pas de LR
                    'batch_size': 32,
                    'optimizer': 'none',
                    'scheduler': 'none',
                    'epochs_requested': 1,
                    'early_stopping_patience': 0,
                    'use_class_weights': False,
                    'gradient_clip': 0.0
                },
                'metadata': {
                    'memory_bank_size': len(self.model.memory_bank) if self.model.memory_bank is not None else 0,
                    'coreset_ratio': self.model.coreset_ratio,
                    'backbone': self.model.backbone_name
                }
            }
            
            logger.info(
                f"PatchCore entraîné avec succès - "
                f"memory_bank_size: {self.history['metadata']['memory_bank_size']}, "
                f"val_accuracy: {val_accuracy:.4f}, "
                f"val_f1: {val_f1:.4f}, "
                f"val_auc: {val_auc:.4f}, "
                f"training_time: {training_time:.1f}s"
            )
            
            return Result.ok(
                {
                    'model': self.model,
                    'preprocessor': self.preprocessor,
                    'history': self.history
                },
                training_time=training_time
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement PatchCore: {e}", exc_info=True)
            return Result.err(f"Entraînement PatchCore échoué: {str(e)}")
    
    def predict(self, X: np.ndarray, **kwargs) -> Result:
        """Wrapper pour prédictions - délègue au trainer interne."""
        if self.model is None or self.preprocessor is None:
            return Result.err("Modèle non entraîné")
        
        logger.info(f"🔮 Prédictions anomalies sur {len(X)} images")
        
        try:
            # Créer un trainer temporaire pour predict
            temp_trainer = ComputerVisionTrainer(
                self.model_config,
                self.training_config
            )
            temp_trainer.model = self.model
            temp_trainer.preprocessor = self.preprocessor
            temp_trainer.device_manager = DeviceManager()
            
            result = temp_trainer.predict(X, **kwargs)
            
            if result.success:
                logger.info("✅ Prédictions anomalies terminées")
            else:
                logger.error(f"❌ Prédictions anomalies échouées: {result.error}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur prédictions anomalies: {e}", exc_info=True)
            return Result.err(f"Prédictions échouées: {str(e)}")


# ==================
# UTILITIES AVANCÉES
# ==================

class ModelInterpreter:
    """Interprétation des prédictions du modèle"""
    
    @staticmethod
    def get_feature_importance(
        model: nn.Module,
        X_sample: np.ndarray,
        preprocessor: DataPreprocessor
    ) -> Result:
        """
        Calcule l'importance des features via gradient.
        
        Args:
            model: Modèle PyTorch
            X_sample: Échantillon à analyser
            preprocessor: Preprocessor du modèle
            
        Returns:
            Result avec carte d'importance
        """
        try:
            model.eval()
            X_norm = preprocessor.transform(X_sample, output_format="channels_first")
            X_tensor = torch.tensor(X_norm, dtype=torch.float32, requires_grad=True)
            
            if torch.cuda.is_available():
                X_tensor = X_tensor.cuda()
                model = model.cuda()
            
            # Forward pass
            output = model(X_tensor)
            pred_class = output.argmax(dim=1)
            
            # Backward pour obtenir gradients
            output[0, pred_class[0]].backward()
            
            # Importance = magnitude du gradient
            importance = X_tensor.grad.abs().cpu().numpy()
            
            return Result.ok({
                'importance_map': importance,
                'predicted_class': int(pred_class[0].cpu().numpy())
            })
            
        except Exception as e:
            logger.error(f"Erreur interprétation: {e}")
            return Result.err(f"Interprétation échouée: {str(e)}")


class DataAugmenter:
    """Augmentation de données pour améliorer la généralisation"""
    
    @staticmethod
    def augment(
        X: np.ndarray,
        y: np.ndarray,
        factor: int = 2,
        methods: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augmente le dataset.
        
        Args:
            X: Images (N, H, W, C)
            y: Labels
            factor: Facteur d'augmentation
            methods: Liste de méthodes ('flip', 'rotate', 'noise')
            
        Returns:
            (X_augmented, y_augmented)
        """
        methods = methods or ['flip', 'rotate']
        
        augmented_X = [X]
        augmented_y = [y]
        
        for _ in range(factor - 1):
            X_aug = X.copy()
            
            if 'flip' in methods and np.random.rand() > 0.5:
                X_aug = np.flip(X_aug, axis=2)  # Flip horizontal
            
            if 'rotate' in methods and np.random.rand() > 0.5:
                angle = np.random.choice([90, 180, 270])
                X_aug = np.rot90(X_aug, k=angle//90, axes=(1, 2))
            
            if 'noise' in methods and np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.01, X_aug.shape)
                X_aug = X_aug + noise
            
            augmented_X.append(X_aug)
            augmented_y.append(y)
        
        return np.concatenate(augmented_X), np.concatenate(augmented_y)


# ========================
# INTÉGRATION AVEC MLFLOW 
# ========================

class MLflowIntegration:
    """
    Intégration avec MLflow pour tracking d'expériences.
    """
    
    def __init__(self, experiment_name: str = "computer_vision_training"):
        self.experiment_name = experiment_name
        self.mlflow_available = False
        
        try:
            import mlflow # type: ignore
            self.mlflow = mlflow
            self.mlflow_available = True
            mlflow.set_experiment(experiment_name)
            logger.info(f"📊 MLflow activé: experiment={experiment_name}")
        except ImportError:
            logger.warning("⚠️ MLflow non disponible, tracking désactivé")
    
    def log_training(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        history: Dict[str, Any],
        test_metrics: Dict[str, Any]
    ) -> None:
        """
        Log une session d'entraînement complète.
        """
        if not self.mlflow_available:
            return
        
        try:
            with self.mlflow.start_run():
                # Log configs
                self.mlflow.log_params({
                    'model_type': model_config.model_type.value,
                    'num_classes': model_config.num_classes,
                    'input_channels': model_config.input_channels,
                    'learning_rate': training_config.learning_rate,
                    'batch_size': training_config.batch_size,
                    'epochs': training_config.epochs,
                    'optimizer': training_config.optimizer.value,
                    'use_class_weights': training_config.use_class_weights
                })
                
                # Log métriques finales
                for metric_name, metric_value in test_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        self.mlflow.log_metric(f"test_{metric_name}", metric_value)
                
                # Log courbes d'entraînement
                train_losses = history.get('train_loss', [])
                val_losses = history.get('val_loss', [])
                
                for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                    self.mlflow.log_metric('train_loss', train_loss, step=epoch)
                    self.mlflow.log_metric('val_loss', val_loss, step=epoch)
                
                logger.info("✅ Métriques loguées dans MLflow")
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur logging MLflow: {e}")


# =====================
# CONFIGURATION FACTORY
# =====================

class ConfigFactory:
    """
    Factory pour créer des configurations pré-définies.
    """
    
    @staticmethod
    def get_config(preset: str) -> Tuple[ModelConfig, TrainingConfig]:
        """
        Retourne des configurations pré-définies.
        
        Presets disponibles:
        - 'quick_test': Entraînement rapide pour tests
        - 'balanced': Configuration équilibrée
        - 'high_accuracy': Optimisé pour précision
        - 'production': Configuration production robuste
        """
        presets = {
            'quick_test': (
                ModelConfig(
                    model_type=ModelType.SIMPLE_CNN,
                    num_classes=2,
                    dropout_rate=0.3
                ),
                TrainingConfig(
                    epochs=5,
                    batch_size=32,
                    learning_rate=1e-3,
                    early_stopping_patience=3,
                    deterministic=True,
                    num_workers=0,
                    pin_memory=False
                )
            ),
            
            'balanced': (
                ModelConfig(
                    model_type=ModelType.CUSTOM_RESNET,
                    num_classes=2,
                    dropout_rate=0.5,
                    base_filters=64
                ),
                TrainingConfig(
                    epochs=50,
                    batch_size=32,
                    learning_rate=1e-4,
                    optimizer=OptimizerType.ADAMW,
                    scheduler=SchedulerType.REDUCE_ON_PLATEAU,
                    early_stopping_patience=10,
                    use_class_weights=True,
                    deterministic=True,
                    num_workers=0,
                    pin_memory=False
                )
            ),
            
            'high_accuracy': (
                ModelConfig(
                    model_type=ModelType.TRANSFER_LEARNING,
                    num_classes=2,
                    pretrained=True,
                    freeze_layers=100,
                    dropout_rate=0.5
                ),
                TrainingConfig(
                    epochs=100,
                    batch_size=16,
                    learning_rate=1e-5,
                    optimizer=OptimizerType.ADAMW,
                    scheduler=SchedulerType.COSINE,
                    early_stopping_patience=20,
                    use_class_weights=True,
                    use_mixed_precision=False,
                    deterministic=True,
                    num_workers=0,
                    pin_memory=False
                )
            ),
            
            'production': (
                ModelConfig(
                    model_type=ModelType.CUSTOM_RESNET,
                    num_classes=2,
                    dropout_rate=0.5,
                    base_filters=64
                ),
                TrainingConfig(
                    epochs=100,
                    batch_size=32,
                    learning_rate=1e-4,
                    optimizer=OptimizerType.ADAMW,
                    scheduler=SchedulerType.REDUCE_ON_PLATEAU,
                    early_stopping_patience=15,
                    reduce_lr_patience=8,
                    use_class_weights=True,
                    gradient_clip=1.0,
                    use_mixed_precision=False,
                    deterministic=True,
                    seed=42,
                    num_workers=0,
                    pin_memory=False
                )
            )
        }
        
        if preset not in presets:
            raise ValueError(
                f"Preset inconnu: {preset}. "
                f"Disponibles: {list(presets.keys())}"
            )
        
        return presets[preset]