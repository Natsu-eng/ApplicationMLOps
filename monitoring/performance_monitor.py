"""
Monitoring des performances et ressources pendant l'entraînement.
"""
import time
import psutil
import logging
from typing import Dict, List, Any, Optional
from src.config.constants import TRAINING_CONSTANTS

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Monitor pour suivre la progression et les ressources pendant l'entraînement."""
    
    def __init__(self):
        self.start_time = None
        self.model_start_time = None
        self.memory_usage = []
        self.current_model = None
        
    def start_training(self) -> None:
        """Démarre le monitoring de l'entraînement."""
        self.start_time = time.time()
        self.memory_usage = []
        logger.info("🚀 Début du monitoring de l'entraînement")
        
    def start_model(self, model_name: str) -> None:
        """Démarre le monitoring pour un modèle spécifique."""
        self.model_start_time = time.time()
        self.current_model = model_name
        logger.info(f"🔧 Début de l'entraînement pour: {model_name}")
        
    def check_resources(self) -> Dict[str, Any]:
        """Vérifie l'utilisation des ressources système."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            resource_info = {
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'cpu_percent': cpu_percent,
                'timestamp': time.time(),
                'model': self.current_model
            }
            
            self.memory_usage.append(resource_info)
            
            # Alertes si utilisation élevée
            high_mem_threshold = TRAINING_CONSTANTS.get("HIGH_MEMORY_THRESHOLD", 85)
            high_cpu_threshold = TRAINING_CONSTANTS.get("HIGH_CPU_THRESHOLD", 90)
            
            if memory.percent > high_mem_threshold:
                logger.warning(f"⚠️ Utilisation mémoire élevée: {memory.percent:.1f}%")
            if cpu_percent > high_cpu_threshold:
                logger.warning(f"⚠️ Utilisation CPU élevée: {cpu_percent:.1f}%")
                
            return resource_info
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification ressources: {e}")
            return {}
    
    def get_model_duration(self) -> float:
        """Retourne la durée d'entraînement du modèle actuel."""
        if self.model_start_time:
            return time.time() - self.model_start_time
        return 0.0
    
    def get_total_duration(self) -> float:
        """Retourne la durée totale d'entraînement."""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé du monitoring."""
        return {
            'total_duration': self.get_total_duration(),
            'memory_samples': len(self.memory_usage),
            'peak_memory': max([m.get('memory_percent', 0) for m in self.memory_usage]) if self.memory_usage else 0,
            'current_model': self.current_model
        }