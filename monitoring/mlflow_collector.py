"""
Collecteur thread-safe pour les runs MLflow.
"""
import threading
from typing import Dict, Any, List
from src.shared.logging import get_logger

logger = get_logger(__name__)

class MLflowRunCollector:
    """Collecteur thread-safe pour les runs MLflow."""
    
    def __init__(self):
        self._runs = []
        self._lock = threading.Lock()
    
    def add_run(self, run_data: Dict[str, Any]) -> None:
        """Ajoute un run de façon thread-safe avec validation."""
        with self._lock:
            if run_data and isinstance(run_data, dict) and run_data.get('run_id'):
                # Validation des données critiques
                required_keys = ['run_id', 'status', 'start_time']
                if all(key in run_data for key in required_keys):
                    self._runs.append(run_data)
                    logger.debug(f"Run MLflow ajouté: {run_data['run_id'][:8]}")
                else:
                    logger.warning(f"Run MLflow incomplet ignoré: {run_data.get('run_id', 'unknown')}")
    
    def get_runs(self) -> List[Dict[str, Any]]:
        """Retourne tous les runs collectés avec validation."""
        with self._lock:
            return [run for run in self._runs if self._is_valid_run(run)]
    
    def _is_valid_run(self, run: Dict) -> bool:
        """Valide la structure d'un run MLflow."""
        return (isinstance(run, dict) and 
                run.get('run_id') and 
                run.get('status') and 
                run.get('start_time'))
    
    def clear(self) -> None:
        """Vide le collecteur."""
        with self._lock:
            self._runs.clear()
    
    def count(self) -> int:
        """Retourne le nombre de runs collectés valides."""
        with self._lock:
            return len([run for run in self._runs if self._is_valid_run(run)])