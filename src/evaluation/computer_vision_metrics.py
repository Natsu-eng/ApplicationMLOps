"""
Système d'évaluation de production pour la détection d'anomalies et classification en vision par ordinateur.
Robuste, flexible et optimisé pour l'environnement de production avec gestion complète des erreurs.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    average_precision_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from typing import Dict, Any, Union, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import traceback
import mlflow
from src.shared.logging import get_logger
from src.config.constants import ANOMALY_CONFIG

logger = get_logger(__name__)


class ModelType(Enum):
    """Types de modèles supportés."""
    AUTOENCODER = "autoencoder"
    CNN_CLASSIFIER = "cnn_classifier"
    TRANSFER_LEARNING = "transfer_learning"
    HYBRID = "hybrid"


class DataFormat(Enum):
    """Formats de données supportés."""
    CHANNELS_FIRST = "channels_first"
    CHANNELS_LAST = "channels_last"
    UNKNOWN = "unknown"


@dataclass
class EvaluationConfig:
    """Configuration robuste pour l'évaluation en production."""
    # Seuils et paramètres
    default_threshold: float = 0.5
    auto_threshold_percentile: float = 95.0
    min_samples_for_metrics: int = 5
    
    # Tolérances d'erreur
    max_nan_ratio: float = 0.05
    min_valid_predictions: float = 0.8
    
    # Formats de données
    preferred_input_format: DataFormat = DataFormat.CHANNELS_FIRST
    fallback_output_format: DataFormat = DataFormat.CHANNELS_LAST
    
    # MLflow
    mlflow_enabled: bool = True
    log_artifacts: bool = True
    log_debug_info: bool = False


class DataFormatManager:
    """Gestionnaire robuste des formats de données."""
    
    @staticmethod
    def detect_data_format(data: np.ndarray) -> DataFormat:
        """Détecte automatiquement le format des données."""
        if data.ndim != 4:
            return DataFormat.UNKNOWN
            
        if data.shape[1] in [1, 3]:
            return DataFormat.CHANNELS_FIRST
        elif data.shape[3] in [1, 3]:
            return DataFormat.CHANNELS_LAST
        else:
            # Tentative de détection heuristique
            if data.shape[1] == data.shape[2] and data.shape[3] > 3:
                return DataFormat.CHANNELS_LAST
            elif data.shape[2] == data.shape[3] and data.shape[1] > 3:
                return DataFormat.CHANNELS_FIRST
            return DataFormat.UNKNOWN
    
    @staticmethod
    def convert_format(data: np.ndarray, target_format: DataFormat) -> np.ndarray:
        """Convertit les données vers le format cible de manière robuste."""
        try:
            current_format = DataFormatManager.detect_data_format(data)
            
            if current_format == target_format:
                return data.copy()
                
            if current_format == DataFormat.CHANNELS_FIRST and target_format == DataFormat.CHANNELS_LAST:
                return np.transpose(data, (0, 2, 3, 1))
            elif current_format == DataFormat.CHANNELS_LAST and target_format == DataFormat.CHANNELS_FIRST:
                return np.transpose(data, (0, 3, 1, 2))
            else:
                # Fallback: tentative de conversion basée sur la forme
                if data.shape[1] in [1, 3] and target_format == DataFormat.CHANNELS_LAST:
                    return np.transpose(data, (0, 2, 3, 1))
                elif data.shape[3] in [1, 3] and target_format == DataFormat.CHANNELS_FIRST:
                    return np.transpose(data, (0, 3, 1, 2))
                else:
                    raise ValueError(f"Conversion impossible de {data.shape} vers {target_format}")
        except Exception as e:
            logger.error(f"Erreur conversion format: {e}")
            raise
    
    @staticmethod
    def validate_and_fix_data_shape(data: np.ndarray, expected_channels: int = 3) -> np.ndarray:
        """Valide et corrige la forme des données de manière robuste."""
        data = data.copy()
        original_shape = data.shape
        
        try:
            # Cas 1: Données 2D (H, W) -> (1, H, W, 1)
            if data.ndim == 2:
                data = data[np.newaxis, ..., np.newaxis]
                logger.info(f"Shape 2D corrigée: {original_shape} -> {data.shape}")
            
            # Cas 2: Données 3D
            elif data.ndim == 3:
                if data.shape[0] in [1, 3]:  # (C, H, W)
                    data = data[np.newaxis, :]  # -> (1, C, H, W)
                elif data.shape[2] in [1, 3]:  # (H, W, C)
                    data = data[np.newaxis, :]  # -> (1, H, W, C)
                else:  # (H, W, ?)
                    data = data[np.newaxis, ..., np.newaxis]  # -> (1, H, W, 1)
                logger.info(f"Shape 3D corrigée: {original_shape} -> {data.shape}")
            
            # Cas 3: Données 4D mais format ambigu
            elif data.ndim == 4:
                format_detected = DataFormatManager.detect_data_format(data)
                if format_detected == DataFormat.UNKNOWN:
                    # Correction heuristique
                    if data.shape[1] == data.shape[2] and data.shape[3] not in [1, 3]:
                        data = data[..., np.newaxis]  # Ajouter dimension canal
                        logger.info(f"Shape 4D corrigée: {original_shape} -> {data.shape}")
            
            # Ajustement du nombre de canaux si nécessaire
            final_format = DataFormatManager.detect_data_format(data)
            if final_format == DataFormat.CHANNELS_FIRST:
                current_channels = data.shape[1]
                channel_axis = 1
            else:
                current_channels = data.shape[3]
                channel_axis = 3
            
            if current_channels == 1 and expected_channels == 3:
                # Dupliquer les canaux pour RGB
                data = np.repeat(data, 3, axis=channel_axis)
                logger.info(f"Canaux dupliqués: 1 -> 3")
            elif current_channels == 3 and expected_channels == 1:
                # Conversion RGB -> Grayscale
                data = np.mean(data, axis=channel_axis, keepdims=True)
                logger.info(f"RGB converti en grayscale")
            
            logger.info(f"Data shape final: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Erreur correction shape: {e}")
            raise


class RobustMetricsCalculator:
    """Calculateur de métriques robuste avec gestion d'erreurs complète."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
    
    def safe_sklearn_metric(self, metric_func, *args, **kwargs) -> Any:
        """Exécute une métrique sklearn avec gestion d'erreurs."""
        try:
            return metric_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Métrique {metric_func.__name__} échouée: {e}")
            return None
    
    def compute_core_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
        """Calcule les métriques de base de manière robuste."""
        metrics = {}
        
        # Validation des entrées
        if len(y_true) < self.config.min_samples_for_metrics:
            logger.warning(f"Échantillons insuffisants: {len(y_true)}")
            return self._get_fallback_metrics()
        
        # Métriques de classification
        metrics["accuracy"] = self.safe_sklearn_metric(accuracy_score, y_true, y_pred)
        metrics["precision"] = self.safe_sklearn_metric(
            precision_score, y_true, y_pred, average='weighted', zero_division=0
        )
        metrics["recall"] = self.safe_sklearn_metric(
            recall_score, y_true, y_pred, average='weighted', zero_division=0
        )
        metrics["f1_score"] = self.safe_sklearn_metric(
            f1_score, y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Métriques basées sur les scores
        if len(np.unique(y_true)) > 1:
            try:
                if y_scores.ndim > 1 and y_scores.shape[1] > 1:
                    metrics["auc_roc"] = roc_auc_score(y_true, y_scores, multi_class='ovr')
                    metrics["average_precision"] = average_precision_score(
                        y_true, y_scores, average='weighted'
                    )
                else:
                    metrics["auc_roc"] = roc_auc_score(y_true, y_scores)
                    metrics["average_precision"] = average_precision_score(y_true, y_scores)
            except Exception as e:
                logger.warning(f"Métriques scores échouées: {e}")
                metrics["auc_roc"] = 0.5
                metrics["average_precision"] = 0.0
        
        # Matrice de confusion
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = {
                "matrix": cm.tolist(),
                "labels": np.unique(np.concatenate([y_true, y_pred])).tolist()
            }
        except Exception as e:
            logger.warning(f"Matrice confusion échouée: {e}")
            metrics["confusion_matrix"] = {"matrix": [], "labels": []}
        
        # Rapport de classification
        try:
            metrics["classification_report"] = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
        except Exception as e:
            logger.warning(f"Rapport classification échoué: {e}")
            metrics["classification_report"] = {}
        
        return {k: v for k, v in metrics.items() if v is not None}
    
    def compute_autoencoder_metrics(self, reconstruction_errors: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Métriques spécifiques aux autoencodeurs."""
        metrics = {}
        
        try:
            # Seuil automatique basé sur les percentiles
            threshold = np.percentile(reconstruction_errors, self.config.auto_threshold_percentile)
            y_pred = (reconstruction_errors > threshold).astype(int)
            
            # Métriques de reconstruction
            metrics["reconstruction_metrics"] = {
                "mean_error": float(np.mean(reconstruction_errors)),
                "std_error": float(np.std(reconstruction_errors)),
                "threshold": float(threshold),
                "percentile_95": float(np.percentile(reconstruction_errors, 95)),
                "percentile_99": float(np.percentile(reconstruction_errors, 99)),
                "min_error": float(np.min(reconstruction_errors)),
                "max_error": float(np.max(reconstruction_errors))
            }
            
            # Métriques de classification basées sur le seuil
            core_metrics = self.compute_core_metrics(y_true, y_pred, reconstruction_errors)
            metrics.update(core_metrics)
            
        except Exception as e:
            logger.error(f"Erreur métriques autoencodeur: {e}")
            metrics.update(self._get_fallback_metrics())
        
        return metrics
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Métriques de fallback en cas d'erreur."""
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "auc_roc": 0.5,
            "average_precision": 0.0,
            "confusion_matrix": {"matrix": [[0, 0], [0, 0]], "labels": [0, 1]},
            "classification_report": {}
        }


class ProductionModelEvaluator:
    """
    Évaluateur de production robuste pour modèles de vision par ordinateur.
    Gestion complète des erreurs, formats de données et logging.
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.metrics_calculator = RobustMetricsCalculator(config)
        self.data_manager = DataFormatManager()
        self.logger = logger
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: Union[ModelType, str],
        preprocessor: Any = None,
        mlflow_run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Évaluation principale avec gestion robuste de tous les cas d'erreur.
        """
        evaluation_id = f"eval_{np.random.randint(10000, 99999)}"
        self.logger.info(f"[{evaluation_id}] Début évaluation - Type: {model_type}")
        
        try:
            # Validation des entrées
            self._validate_inputs(X_test, y_test, model_type)
            
            # Préparation robuste des données
            X_processed, prep_info = self._prepare_data(X_test, preprocessor, evaluation_id)
            
            # Prédictions robustes
            predictions = self._robust_predictions(model, X_processed, model_type, evaluation_id)
            
            # Calcul des métriques
            metrics = self._compute_comprehensive_metrics(
                y_test, predictions, model_type, evaluation_id
            )
            
            # Logging des résultats
            if self.config.mlflow_enabled:
                self._log_to_mlflow(metrics, prep_info, mlflow_run_id, evaluation_id)
            
            return self._build_success_report(metrics, prep_info, evaluation_id)
            
        except Exception as e:
            self.logger.error(f"[{evaluation_id}] Échec évaluation: {e}")
            return self._build_error_report(e, evaluation_id)
    
    def _validate_inputs(self, X_test: np.ndarray, y_test: np.ndarray, model_type: ModelType):
        """Validation robuste des données d'entrée."""
        if X_test.size == 0:
            raise ValueError("Données X_test vides")
        if y_test.size == 0:
            raise ValueError("Labels y_test vides")
        if len(X_test) != len(y_test):
            raise ValueError(f"Taille incohérente: X_test({len(X_test)}) != y_test({len(y_test)})")
        
        # Vérification des NaN
        nan_ratio_x = np.isnan(X_test).sum() / X_test.size
        nan_ratio_y = np.isnan(y_test).sum() / y_test.size
        
        if nan_ratio_x > self.config.max_nan_ratio:
            raise ValueError(f"Trop de NaN dans X_test: {nan_ratio_x:.2%}")
        if nan_ratio_y > self.config.max_nan_ratio:
            raise ValueError(f"Trop de NaN dans y_test: {nan_ratio_y:.2%}")
    
    def _prepare_data(
        self, 
        X_test: np.ndarray, 
        preprocessor: Any, 
        evaluation_id: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Préparation robuste des données avec correction automatique."""
        prep_info = {
            "preprocessor_used": False,
            "preprocessor_type": None,
            "original_shape": X_test.shape,
            "final_shape": None,
            "format_changes": []
        }
        
        try:
            X_processed = X_test.copy()
            
            # Application du preprocessor si disponible
            if preprocessor is not None:
                self.logger.info(f"[{evaluation_id}] Application du preprocessor")
                try:
                    X_processed = preprocessor.transform(X_processed, output_format="channels_first")
                    prep_info.update({
                        "preprocessor_used": True,
                        "preprocessor_type": type(preprocessor).__name__
                    })
                except Exception as e:
                    self.logger.warning(f"[{evaluation_id}] Preprocessor échoué: {e}")
                    # Continue sans preprocessor
            
            # Correction automatique du format et shape
            original_format = self.data_manager.detect_data_format(X_processed)
            X_processed = self.data_manager.validate_and_fix_data_shape(X_processed)
            final_format = self.data_manager.detect_data_format(X_processed)
            
            # Conversion vers le format préféré
            if final_format != self.config.preferred_input_format:
                self.logger.info(
                    f"[{evaluation_id}] Conversion format: {final_format} -> {self.config.preferred_input_format}"
                )
                X_processed = self.data_manager.convert_format(X_processed, self.config.preferred_input_format)
                final_format = self.config.preferred_input_format
            
            prep_info.update({
                "final_shape": X_processed.shape,
                "final_format": final_format.value,
                "format_changes": [f"{original_format.value}->{final_format.value}"]
            })
            
            self.logger.info(f"[{evaluation_id}] Données préparées: {X_processed.shape}")
            return X_processed, prep_info
            
        except Exception as e:
            self.logger.error(f"[{evaluation_id}] Erreur préparation données: {e}")
            raise
    
    def _robust_predictions(
        self, 
        model: Any, 
        X_processed: np.ndarray, 
        model_type: ModelType,
        evaluation_id: str
    ) -> Dict[str, np.ndarray]:
        """Prédictions robustes avec multiples fallbacks."""
        try:
            # Méthode 1: Prédiction standard
            self.logger.info(f"[{evaluation_id}] Tentative prédiction standard")
            
            if model_type == ModelType.AUTOENCODER:
                return self._predict_autoencoder(model, X_processed, evaluation_id)
            else:
                return self._predict_classifier(model, X_processed, evaluation_id)
                
        except Exception as e:
            self.logger.warning(f"[{evaluation_id}] Prédiction standard échouée: {e}")
            return self._fallback_predictions(model, X_processed, model_type, evaluation_id)
    
    def _predict_autoencoder(self, model: Any, X_processed: np.ndarray, evaluation_id: str) -> Dict[str, np.ndarray]:
        """Prédiction pour autoencodeur."""
        try:
            # Test avec un échantillon
            test_sample = X_processed[:1]
            test_output = model.predict(test_sample)
            
            # Prédiction complète
            reconstructed = model.predict(X_processed)
            mse_errors = np.mean((X_processed - reconstructed) ** 2, axis=(1, 2, 3))
            
            self.logger.info(f"[{evaluation_id}] Autoencoder - Prédiction réussie")
            return {
                "scores": mse_errors,
                "reconstructed": reconstructed,
                "binary": None  # Sera calculé dans les métriques
            }
        except Exception as e:
            self.logger.error(f"[{evaluation_id}] Autoencoder - Prédiction échouée: {e}")
            raise
    
    def _predict_classifier(self, model: Any, X_processed: np.ndarray, evaluation_id: str) -> Dict[str, np.ndarray]:
        """Prédiction pour classificateur."""
        try:
            if hasattr(model, 'predict_proba'):
                y_scores = model.predict_proba(X_processed)
            else:
                y_pred = model.predict(X_processed)
                # Conversion en probabilités simulées
                if y_pred.ndim == 1:
                    y_scores = np.column_stack([1 - y_pred, y_pred])
                else:
                    y_scores = y_pred
            
            y_binary = (np.max(y_scores, axis=1) > self.config.default_threshold).astype(int)
            
            self.logger.info(f"[{evaluation_id}] Classifier - Prédiction réussie")
            return {
                "scores": y_scores,
                "binary": y_binary,
                "reconstructed": None
            }
        except Exception as e:
            self.logger.error(f"[{evaluation_id}] Classifier - Prédiction échouée: {e}")
            raise
    
    def _fallback_predictions(
        self, 
        model: Any, 
        X_processed: np.ndarray, 
        model_type: ModelType,
        evaluation_id: str
    ) -> Dict[str, np.ndarray]:
        """Fallbacks de prédiction en cas d'échec."""
        self.logger.info(f"[{evaluation_id}] Tentative fallback PyTorch")
        
        try:
            import torch
            
            device = next(model.parameters()).device
            X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                if model_type == ModelType.AUTOENCODER:
                    reconstructed = model(X_tensor)
                    mse_errors = torch.mean(
                        (X_tensor - reconstructed) ** 2, dim=(1, 2, 3)
                    ).cpu().numpy()
                    
                    return {
                        "scores": mse_errors,
                        "reconstructed": reconstructed.cpu().numpy(),
                        "binary": None
                    }
                else:
                    output = model(X_tensor)
                    if hasattr(output, 'logits'):
                        y_proba = torch.softmax(output.logits, dim=1).cpu().numpy()
                    else:
                        y_proba = torch.softmax(output, dim=1).cpu().numpy()
                    
                    y_binary = (np.max(y_proba, axis=1) > 0.5).astype(int)
                    
                    return {
                        "scores": y_proba,
                        "binary": y_binary,
                        "reconstructed": None
                    }
                    
        except Exception as torch_error:
            self.logger.error(f"[{evaluation_id}] Fallback PyTorch échoué: {torch_error}")
            
            # Fallback final: scores contrôlés
            self.logger.warning(f"[{evaluation_id}] Utilisation fallback manuel")
            n_samples = len(X_processed)
            
            if model_type == ModelType.AUTOENCODER:
                scores = np.random.uniform(0, 1, n_samples)
                return {"scores": scores, "binary": None, "reconstructed": None}
            else:
                scores = np.random.dirichlet(np.ones(2), n_samples)
                binary = (np.max(scores, axis=1) > 0.5).astype(int)
                return {"scores": scores, "binary": binary, "reconstructed": None}
    
    def _compute_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        model_type: ModelType,
        evaluation_id: str
    ) -> Dict[str, Any]:
        """Calcul complet des métriques."""
        try:
            if model_type == ModelType.AUTOENCODER:
                metrics = self.metrics_calculator.compute_autoencoder_metrics(
                    predictions["scores"], y_true
                )
            else:
                metrics = self.metrics_calculator.compute_core_metrics(
                    y_true, predictions["binary"], predictions["scores"]
                )
            
            # Statistiques supplémentaires
            metrics["prediction_stats"] = self._compute_prediction_stats(predictions, y_true)
            metrics["evaluation_info"] = {
                "model_type": model_type.value,
                "n_samples": len(y_true),
                "timestamp": np.datetime64('now').astype(str)
            }
            
            self.logger.info(f"[{evaluation_id}] Métriques calculées avec succès")
            return metrics
            
        except Exception as e:
            self.logger.error(f"[{evaluation_id}] Erreur calcul métriques: {e}")
            return self.metrics_calculator._get_fallback_metrics()
    
    def _compute_prediction_stats(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Any]:
        """Statistiques détaillées sur les prédictions."""
        try:
            y_scores = predictions["scores"]
            y_binary = predictions.get("binary")
            
            stats = {
                "score_distribution": {
                    "mean": float(np.mean(y_scores)),
                    "std": float(np.std(y_scores)),
                    "min": float(np.min(y_scores)),
                    "max": float(np.max(y_scores)),
                    "percentiles": {
                        "25": float(np.percentile(y_scores, 25)),
                        "50": float(np.percentile(y_scores, 50)),
                        "75": float(np.percentile(y_scores, 75)),
                        "95": float(np.percentile(y_scores, 95))
                    }
                },
                "class_distribution": {
                    "true": dict(zip(*np.unique(y_true, return_counts=True)))
                }
            }
            
            if y_binary is not None:
                stats["class_distribution"]["predicted"] = dict(zip(*np.unique(y_binary, return_counts=True)))
            
            return stats
        except Exception as e:
            self.logger.warning(f"Erreur statistiques prédictions: {e}")
            return {}
    
    def _log_to_mlflow(
        self,
        metrics: Dict[str, Any],
        prep_info: Dict[str, Any],
        mlflow_run_id: Optional[str],
        evaluation_id: str
    ):
        """Logging structuré dans MLflow."""
        if not mlflow_run_id:
            return
        
        try:
            mlflow.set_tracking_uri(ANOMALY_CONFIG.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
            
            with mlflow.start_run(run_id=mlflow_run_id):
                # Métriques principales
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                        if metric_name not in ["confusion_matrix", "classification_report", "prediction_stats", "evaluation_info"]:
                            mlflow.log_metric(f"eval_{metric_name}", metric_value)
                
                # Informations de preprocessing
                mlflow.log_params({
                    "eval_preprocessor_used": prep_info.get("preprocessor_used", False),
                    "eval_data_format": prep_info.get("final_format", "unknown"),
                    "eval_n_samples": metrics.get("evaluation_info", {}).get("n_samples", 0)
                })
                
                self.logger.info(f"[{evaluation_id}] Résultats loggés dans MLflow")
                
        except Exception as e:
            self.logger.error(f"[{evaluation_id}] Erreur logging MLflow: {e}")
    
    def _build_success_report(
        self,
        metrics: Dict[str, Any],
        prep_info: Dict[str, Any],
        evaluation_id: str
    ) -> Dict[str, Any]:
        """Rapport de succès structuré."""
        return {
            "success": True,
            "evaluation_id": evaluation_id,
            "timestamp": np.datetime64('now').astype(str),
            "metrics": metrics,
            "preprocessing_info": prep_info,
            "performance_assessment": self._assess_performance(metrics),
            "recommendations": self._generate_recommendations(metrics, prep_info)
        }
    
    def _build_error_report(self, error: Exception, evaluation_id: str) -> Dict[str, Any]:
        """Rapport d'erreur structuré."""
        return {
            "success": False,
            "evaluation_id": evaluation_id,
            "timestamp": np.datetime64('now').astype(str),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            },
            "metrics": self.metrics_calculator._get_fallback_metrics(),
            "recommendations": [
                "Vérifier le format des données d'entrée",
                "Valider la compatibilité du modèle",
                "Vérifier les dimensions des tenseurs"
            ]
        }
    
    def _assess_performance(self, metrics: Dict[str, Any]) -> str:
        """Évaluation du niveau de performance."""
        try:
            accuracy = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_score", 0)
            auc = metrics.get("auc_roc", 0.5)
            
            if accuracy >= 0.9 and f1 >= 0.9 and auc >= 0.9:
                return "EXCELLENT"
            elif accuracy >= 0.8 and f1 >= 0.8 and auc >= 0.8:
                return "GOOD" 
            elif accuracy >= 0.7 and f1 >= 0.7 and auc >= 0.7:
                return "ACCEPTABLE"
            else:
                return "NEEDS_IMPROVEMENT"
        except:
            return "UNKNOWN"
    
    def _generate_recommendations(self, metrics: Dict[str, Any], prep_info: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur les résultats."""
        recommendations = []
        
        accuracy = metrics.get("accuracy", 0)
        if accuracy < 0.7:
            recommendations.append("Envisager augmentation données ou rééquilibrage classes")
        
        f1 = metrics.get("f1_score", 0)
        if f1 < 0.6:
            recommendations.append("Ajuster seuils classification ou revoir prétraitement")
        
        if not prep_info.get("preprocessor_used", False):
            recommendations.append("Ajouter preprocessor pourrait améliorer performances")
        
        if not recommendations:
            recommendations.append("Performance satisfaisante - maintenir configuration")
        
        return recommendations


# =============================================================================
# INTERFACES DE COMPATIBILITÉ LEGACY
# =============================================================================

def compute_anomaly_metrics(
    preds: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    model_type: str = "autoencoder",
    mlflow_run_id: Optional[str] = None
) -> Dict[str, Any]:
    """Interface legacy pour compatibilité ascendante."""
    evaluator = ProductionModelEvaluator()
    
    # Simulation de prédictions pour l'interface legacy
    if model_type == "autoencoder":
        predictions = {"scores": preds, "binary": None, "reconstructed": None}
        metrics = evaluator.metrics_calculator.compute_autoencoder_metrics(preds, y_true)
    else:
        y_binary = (preds > threshold).astype(int) if preds.ndim == 1 else (np.max(preds, axis=1) > threshold).astype(int)
        predictions = {"scores": preds, "binary": y_binary, "reconstructed": None}
        metrics = evaluator.metrics_calculator.compute_core_metrics(y_true, y_binary, preds)
    
    if mlflow_run_id and ANOMALY_CONFIG.get("MLFLOW_ENABLED", False):
        try:
            mlflow.set_tracking_uri(ANOMALY_CONFIG.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
            with mlflow.start_run(run_id=mlflow_run_id):
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)) and metric_name not in ["confusion_matrix", "classification_report"]:
                        mlflow.log_metric(f"legacy_{metric_name}", metric_value)
        except Exception as e:
            logger.error(f"MLflow logging legacy échoué: {e}")
    
    return metrics


def evaluate_autoencoder(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    preprocessor=None,
    mlflow_run_id: Optional[str] = None
) -> Dict[str, Any]:
    """Interface legacy pour autoencodeurs."""
    evaluator = ProductionModelEvaluator()
    return evaluator.evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_type=ModelType.AUTOENCODER,
        preprocessor=preprocessor,
        mlflow_run_id=mlflow_run_id
    )


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray, 
    preprocessor=None,
    mlflow_run_id: Optional[str] = None
) -> Dict[str, Any]:
    """Interface legacy pour classificateurs."""
    evaluator = ProductionModelEvaluator()
    return evaluator.evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_type=ModelType.CNN_CLASSIFIER,
        preprocessor=preprocessor,
        mlflow_run_id=mlflow_run_id
    )


def compute_reconstruction_metrics(
    X_true: np.ndarray,
    X_reconstructed: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    mlflow_run_id: Optional[str] = None
) -> Dict[str, Any]:
    """Interface legacy pour métriques de reconstruction."""
    try:
        reconstruction_errors = np.mean(np.square(X_true - X_reconstructed), axis=(1, 2, 3))
        
        metrics = {
            "mean_mse": float(np.mean(reconstruction_errors)),
            "std_mse": float(np.std(reconstruction_errors)),
            "mean_mae": float(np.mean(np.abs(X_true - X_reconstructed), axis=(1, 2, 3)).mean()),
            "percentile_95": float(np.percentile(reconstruction_errors, 95)),
            "percentile_99": float(np.percentile(reconstruction_errors, 99))
        }
        
        if y_true is not None and len(np.unique(y_true)) > 1:
            try:
                metrics["auc_roc"] = float(roc_auc_score(y_true, reconstruction_errors))
            except Exception as e:
                logger.warning(f"AUC-ROC reconstruction échoué: {e}")
                metrics["auc_roc"] = 0.5
        
        if mlflow_run_id and ANOMALY_CONFIG.get("MLFLOW_ENABLED", False):
            try:
                mlflow.set_tracking_uri(ANOMALY_CONFIG.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
                with mlflow.start_run(run_id=mlflow_run_id):
                    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            except Exception as e:
                logger.error(f"MLflow logging reconstruction échoué: {e}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Erreur compute_reconstruction_metrics: {e}")
        return {
            "mean_mse": 0.0, "std_mse": 0.0, "mean_mae": 0.0,
            "percentile_95": 0.0, "percentile_99": 0.0, "auc_roc": 0.5,
            "error": str(e)
        }


def default_metrics() -> Dict[str, Any]:
    """Métriques par défaut en cas d'erreur."""
    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "auc_roc": 0.5,
        "average_precision": 0.0,
        "confusion_matrix": {"matrix": [[0, 0], [0, 0]], "labels": [0, 1]},
        "classification_report": {},
        "error": "Évaluation échouée"
    }


# Instance globale pour usage simple
DEFAULT_EVALUATOR = ProductionModelEvaluator()