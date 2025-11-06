"""
Gestionnaire d'état global pour l'entraînement - Thread-safe.
"""
import threading
from typing import Dict, Any, List
from contextlib import contextmanager
from monitoring.mlflow_collector import MLflowRunCollector
from src.shared.logging import get_logger

logger = get_logger(__name__)

class TrainingStateManager:
    """Gestionnaire d'état global pour l'entraînement."""
    
    def __init__(self):
        self.mlflow_collector = MLflowRunCollector()
        self._training_lock = threading.Lock()
        self._active_training = False
    
    @contextmanager
    def training_session(self):
        """Context manager pour une session d'entraînement."""
        with self._training_lock:
            self._active_training = True
            self.mlflow_collector.clear()
            try:
                yield self
            finally:
                self._active_training = False
    
    def is_training_active(self) -> bool:
        """Vérifie si un entraînement est en cours."""
        return self._active_training

# Instance globale
TRAINING_STATE = TrainingStateManager()