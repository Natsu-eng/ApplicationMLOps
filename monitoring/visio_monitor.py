from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.base import accuracy_score
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

from src.data.computer_vision_preprocessing import DataPreprocessor, Result
from src.shared.logging import get_logger

logger = get_logger(__name__)
# ========================
# MONITORING EN PRODUCTION
# ========================

class ProductionMonitor:
    """
    Monitoring des modèles en production.
    
    Features:
    - Détection de drift
    - Alertes de performance
    - Métriques business
    """
    
    def __init__(
        self,
        model: nn.Module,
        preprocessor: DataPreprocessor,
        baseline_metrics: Dict[str, float],
        alert_thresholds: Dict[str, float] = None
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.baseline_metrics = baseline_metrics
        self.alert_thresholds = alert_thresholds or {
            'accuracy_drop': 0.05,  # 5% drop
            'f1_drop': 0.05
        }
        
        self.predictions_log = []
        self.metrics_history = []
        
        logger.info("ProductionMonitor initialisé")
    
    def predict_and_monitor(
        self,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ) -> Result:
        """
        Prédiction avec monitoring.
        
        Args:
            X: Features
            y_true: Labels vrais (optionnel, pour calcul métriques)
        """
        try:
            # Preprocessing
            X_norm = self.preprocessor.transform(X)
            
            # Prédiction
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_norm, dtype=torch.float32)
                if torch.cuda.is_available():
                    X_tensor = X_tensor.cuda()
                    self.model = self.model.cuda()
                
                output = self.model(X_tensor)
                predictions = output.argmax(dim=1).cpu().numpy()
                probabilities = torch.softmax(output, dim=1).cpu().numpy()
            
            # Logging
            self.predictions_log.append({
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(X),
                'predictions': predictions.tolist(),
                'mean_confidence': float(probabilities.max(axis=1).mean())
            })
            
            result_data = {
                'predictions': predictions,
                'probabilities': probabilities
            }
            
            # Calcul métriques si labels fournis
            if y_true is not None:
                metrics = {
                    'accuracy': float(accuracy_score(y_true, predictions)),
                    'f1': float(f1_score(y_true, predictions, average='weighted', zero_division=0))
                }
                
                # Détection d'alertes
                alerts = self._check_alerts(metrics)
                
                self.metrics_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics,
                    'alerts': alerts
                })
                
                result_data['metrics'] = metrics
                result_data['alerts'] = alerts
                
                if alerts:
                    logger.warning(f"Alertes détectées", alerts=alerts)
            
            return Result.ok(result_data)
            
        except Exception as e:
            logger.error(f"Erreur monitoring: {e}", exc_info=True)
            return Result.err(f"Monitoring échoué: {str(e)}")
    
    def _check_alerts(self, current_metrics: Dict[str, float]) -> List[str]:
        """Vérifie les alertes de performance"""
        alerts = []
        
        for metric, current_value in current_metrics.items():
            baseline_value = self.baseline_metrics.get(metric)
            if baseline_value is None:
                continue
            
            drop = baseline_value - current_value
            threshold = self.alert_thresholds.get(f'{metric}_drop', 0.05)
            
            if drop > threshold:
                alerts.append(
                    f"{metric} a chuté de {drop:.2%} "
                    f"(baseline: {baseline_value:.4f}, current: {current_value:.4f})"
                )
        
        return alerts
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Génère un rapport de monitoring"""
        if not self.predictions_log:
            return {"error": "Aucune prédiction enregistrée"}
        
        report = {
            'total_predictions': sum(log['n_samples'] for log in self.predictions_log),
            'n_batches': len(self.predictions_log),
            'time_range': {
                'start': self.predictions_log[0]['timestamp'],
                'end': self.predictions_log[-1]['timestamp']
            }
        }
        
        if self.metrics_history:
            recent_metrics = self.metrics_history[-10:]  # 10 derniers
            report['recent_performance'] = {
                'accuracy': {
                    'mean': float(np.mean([m['metrics']['accuracy'] for m in recent_metrics])),
                    'std': float(np.std([m['metrics']['accuracy'] for m in recent_metrics]))
                },
                'f1': {
                    'mean': float(np.mean([m['metrics']['f1'] for m in recent_metrics])),
                    'std': float(np.std([m['metrics']['f1'] for m in recent_metrics]))
                }
            }
            
            all_alerts = [alert for m in self.metrics_history for alert in m['alerts']]
            report['total_alerts'] = len(all_alerts)
            report['unique_alerts'] = list(set(all_alerts))
        
        return report