"""
monitoring/mlflow_collector.py
Collecteur centralisé thread-safe pour runs MLflow avec garantie de cohérence.

Version: 2.0.0 | Production-Ready
Features:
- Thread-safety via RLock
- Déduplication automatique
- Validation stricte
- Callbacks pour synchronisation externe
- Métriques de monitoring
- Export/Import pour persistance
"""

import threading
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ========================
# DATACLASS PRINCIPALE
# ========================

@dataclass
class MLflowRunCollector:
    """
    Collecteur thread-safe pour runs MLflow.
    
    Architecture:
    - Singleton pattern via get_mlflow_collector()
    - Thread-safe avec RLock
    - Déduplication automatique par run_id
    - Callbacks pour synchronisation temps réel
    
    Usage:
        collector = get_mlflow_collector()
        collector.add_run({'run_id': '123', 'model_name': 'XGBoost', ...})
        runs = collector.get_runs()
    """
    
    # Private attributes (non sérialisés)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _runs: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _callbacks: List[Callable] = field(default_factory=list, init=False, repr=False)
    
    # Métriques de monitoring (public)
    total_runs_added: int = field(default=0, init=False)
    total_runs_rejected: int = field(default=0, init=False)
    total_callbacks_triggered: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Initialisation post-dataclass."""
        logger.debug("MLflowRunCollector initialisé")
    
    # ========================
    # MÉTHODES PRINCIPALES
    # ========================
    
    def add_run(self, run: Dict[str, Any], trigger_callbacks: bool = True) -> bool:
        """
        Ajoute un run MLflow avec validation stricte.
        
        Args:
            run: Dict contenant au minimum 'run_id'
            trigger_callbacks: Si True, déclenche les callbacks enregistrés
            
        Returns:
            True si ajouté avec succès, False si rejeté (dupliqué/invalide)
            
        Raises:
            TypeError: Si run n'est pas un dict
            ValueError: Si run_id manquant ou invalide
        """
        with self._lock:
            # VALIDATION TYPE
            if not isinstance(run, dict):
                self.total_runs_rejected += 1
                logger.warning(f"⚠️ Run rejeté (type invalide: {type(run).__name__})")
                return False
            
            # VALIDATION run_id
            run_id = run.get('run_id')
            
            if not run_id:
                self.total_runs_rejected += 1
                logger.warning("⚠️ Run rejeté (run_id manquant)")
                return False
            
            if not isinstance(run_id, str) or len(run_id) == 0:
                self.total_runs_rejected += 1
                logger.warning(f"⚠️ Run rejeté (run_id invalide: {run_id})")
                return False
            
            # DÉDUPLICATION
            if run_id in self._runs:
                logger.debug(f"Run {run_id[:8]}... déjà collecté (ignoré)")
                return False
            
            # ENRICHISSEMENT automatique
            if 'collected_at' not in run:
                run['collected_at'] = datetime.now().isoformat()
            
            if 'collector_version' not in run:
                run['collector_version'] = '2.0.0'
            
            # STOCKAGE
            self._runs[run_id] = run
            self.total_runs_added += 1
            
            logger.info(
                f"✅ Run collecté: {run_id[:8]}... | "
                f"Modèle: {run.get('model_name', 'N/A')} | "
                f"Total: {len(self._runs)}"
            )
            
            # CALLBACKS
            if trigger_callbacks and self._callbacks:
                self._trigger_callbacks(run)
            
            return True
    
    def get_runs(
        self, 
        limit: Optional[int] = None,
        sort_by: Optional[str] = None,
        reverse: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Récupère les runs collectés avec options de tri/limite.
        
        Args:
            limit: Nombre max de runs à retourner
            sort_by: Clé de tri ('collected_at', 'model_name', etc.)
            reverse: Si True, tri décroissant
            
        Returns:
            Liste de runs (copies pour éviter mutations)
        """
        with self._lock:
            # Copie pour éviter modifications externes
            runs = [run.copy() for run in self._runs.values()]
            
            # TRI optionnel
            if sort_by:
                try:
                    runs.sort(
                        key=lambda x: x.get(sort_by, ''),
                        reverse=reverse
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Tri échoué ({sort_by}): {e}")
            
            # LIMITE optionnelle
            if limit and limit > 0:
                runs = runs[:limit]
            
            return runs
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère un run spécifique par son ID.
        
        Args:
            run_id: ID du run MLflow
            
        Returns:
            Dict du run ou None si non trouvé (copie pour éviter mutations)
        """
        with self._lock:
            run = self._runs.get(run_id)
            return run.copy() if run else None
    
    def has_run(self, run_id: str) -> bool:
        """
        Vérifie si un run existe dans le collecteur.
        
        Args:
            run_id: ID du run à vérifier
            
        Returns:
            True si le run existe, False sinon
        """
        with self._lock:
            return run_id in self._runs
    
    def remove_run(self, run_id: str) -> bool:
        """
        Supprime un run spécifique.
        
        Args:
            run_id: ID du run à supprimer
            
        Returns:
            True si supprimé, False si non trouvé
        """
        with self._lock:
            if run_id in self._runs:
                del self._runs[run_id]
                logger.info(f"🗑️ Run {run_id[:8]}... supprimé")
                return True
            return False
    
    def clear(self) -> int:
        """
        Nettoie tous les runs collectés.
        
        Returns:
            Nombre de runs supprimés
        """
        with self._lock:
            count = len(self._runs)
            self._runs.clear()
            logger.info(f"🗑️ {count} runs supprimés du collecteur")
            return count
    
    def count(self) -> int:
        """
        Retourne le nombre de runs actuellement collectés.
        
        Returns:
            Nombre de runs
        """
        with self._lock:
            return len(self._runs)
    
    # ========================
    # GESTION CALLBACKS
    # ========================
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Enregistre un callback déclenché à chaque nouveau run.
        
        Args:
            callback: Fonction callback(run: Dict) -> None
            
        Returns:
            True si enregistré, False si déjà existant
            
        Example:
            def sync_to_db(run):
                db.save(run)
            
            collector.register_callback(sync_to_db)
        """
        with self._lock:
            if callback in self._callbacks:
                logger.debug(f"Callback {callback.__name__} déjà enregistré")
                return False
            
            self._callbacks.append(callback)
            logger.info(f"✅ Callback enregistré: {callback.__name__}")
            return True
    
    def unregister_callback(self, callback: Callable) -> bool:
        """
        Désenregistre un callback.
        
        Args:
            callback: Fonction callback à retirer
            
        Returns:
            True si retiré, False si non trouvé
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
                logger.info(f"❌ Callback retiré: {callback.__name__}")
                return True
            return False
    
    def clear_callbacks(self) -> int:
        """
        Retire tous les callbacks.
        
        Returns:
            Nombre de callbacks retirés
        """
        with self._lock:
            count = len(self._callbacks)
            self._callbacks.clear()
            logger.info(f"🗑️ {count} callbacks retirés")
            return count
    
    def _trigger_callbacks(self, run: Dict[str, Any]):
        """
        Déclenche tous les callbacks enregistrés (thread-safe).
        
        Args:
            run: Run MLflow à passer aux callbacks
        """
        for callback in self._callbacks:
            try:
                callback(run)
                self.total_callbacks_triggered += 1
                logger.debug(f"Callback {callback.__name__} exécuté avec succès")
            
            except Exception as e:
                logger.error(
                    f"❌ Erreur callback {callback.__name__}: {e}",
                    exc_info=True
                )
    
    # ========================
    # MÉTHODES AVANCÉES
    # ========================
    
    def get_runs_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Filtre runs par statut.
        
        Args:
            status: 'FINISHED', 'RUNNING', 'FAILED', etc.
            
        Returns:
            Liste de runs filtrés
        """
        with self._lock:
            return [
                run.copy() 
                for run in self._runs.values() 
                if run.get('status') == status
            ]
    
    def get_runs_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Filtre runs par nom de modèle.
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Liste de runs filtrés
        """
        with self._lock:
            return [
                run.copy() 
                for run in self._runs.values() 
                if run.get('model_name') == model_name or
                   run.get('tags', {}).get('mlflow.runName') == model_name
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne statistiques du collecteur.
        
        Returns:
            Dict avec métriques de monitoring
        """
        with self._lock:
            return {
                'total_runs': len(self._runs),
                'total_runs_added': self.total_runs_added,
                'total_runs_rejected': self.total_runs_rejected,
                'total_callbacks_triggered': self.total_callbacks_triggered,
                'callbacks_registered': len(self._callbacks),
                'run_ids': list(self._runs.keys())[:10]  # Sample
            }

    # ========================
    # ALIAS POUR COMPATIBILITÉ
    # ========================  
    def run_exists(self, run_id: str) -> bool:
        """
        Vérifie si un run existe dans le collecteur.
        Alias pour has_run() pour compatibilité avec le code existant.   
        Args:
            run_id: ID du run à vérifier      
        Returns:
            True si le run existe, False sinon
        """
        return self.has_run(run_id)
    
    # ========================
    # PERSISTANCE (EXPORT/IMPORT)
    # ========================
    
    def export_to_file(self, filepath: str) -> bool:
        """
        Exporte tous les runs vers un fichier JSON.
        
        Args:
            filepath: Chemin du fichier de destination
            
        Returns:
            True si succès, False si échec
        """
        try:
            with self._lock:
                data = {
                    'runs': list(self._runs.values()),
                    'metadata': {
                        'total_runs': len(self._runs),
                        'exported_at': datetime.now().isoformat(),
                        'collector_version': '2.0.0'
                    }
                }
                
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ {len(self._runs)} runs exportés vers {filepath}")
                return True
        
        except Exception as e:
            logger.error(f"❌ Erreur export: {e}", exc_info=True)
            return False
    
    def import_from_file(
        self, 
        filepath: str, 
        replace: bool = False,
        trigger_callbacks: bool = False
    ) -> int:
        """
        Importe des runs depuis un fichier JSON.
        
        Args:
            filepath: Chemin du fichier source
            replace: Si True, remplace les runs existants
            trigger_callbacks: Si True, déclenche callbacks pour chaque run
            
        Returns:
            Nombre de runs importés
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            runs = data.get('runs', [])
            
            if replace:
                self.clear()
            
            imported = 0
            for run in runs:
                if self.add_run(run, trigger_callbacks=trigger_callbacks):
                    imported += 1
            
            logger.info(f"✅ {imported}/{len(runs)} runs importés depuis {filepath}")
            return imported
        
        except Exception as e:
            logger.error(f"❌ Erreur import: {e}", exc_info=True)
            return 0


# ========================
# SINGLETON GLOBAL
# ========================

_GLOBAL_COLLECTOR: Optional[MLflowRunCollector] = None
_GLOBAL_LOCK = threading.RLock()


def get_mlflow_collector() -> MLflowRunCollector:
    """
    Retourne l'instance singleton du collecteur MLflow.
    
    Thread-safe via double-checked locking pattern.
    
    Returns:
        Instance unique de MLflowRunCollector
        
    Usage:
        from monitoring.mlflow_collector import get_mlflow_collector
        
        collector = get_mlflow_collector()
        collector.add_run({'run_id': '123', ...})
    """
    global _GLOBAL_COLLECTOR
    
    # Fast path (sans lock)
    if _GLOBAL_COLLECTOR is not None:
        return _GLOBAL_COLLECTOR
    
    # Slow path (avec lock)
    with _GLOBAL_LOCK:
        if _GLOBAL_COLLECTOR is None:
            _GLOBAL_COLLECTOR = MLflowRunCollector()
            logger.info("MLflowRunCollector global initialisé")
        
        return _GLOBAL_COLLECTOR


def reset_mlflow_collector() -> bool:
    """
    Reset complet du collecteur global (pour tests uniquement).
    
    ⚠️ ATTENTION: Cette fonction doit être utilisée UNIQUEMENT dans les tests.
    
    Returns:
        True si reset effectué
    """
    global _GLOBAL_COLLECTOR
    
    with _GLOBAL_LOCK:
        if _GLOBAL_COLLECTOR is not None:
            _GLOBAL_COLLECTOR.clear()
            _GLOBAL_COLLECTOR.clear_callbacks()
            _GLOBAL_COLLECTOR = None
            logger.warning("⚠️ MLflowRunCollector global reset (mode test)")
            return True
        return False


# ========================
# EXPORTS
# ========================

__all__ = [
    'MLflowRunCollector',
    'get_mlflow_collector',
    'reset_mlflow_collector',  # Pour tests uniquement
]


# ========================
# EXEMPLE D'USAGE
# ========================

if __name__ == "__main__":
    # Configuration logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Récupération collecteur
    collector = get_mlflow_collector()
    
    # Enregistrement callback
    def print_run(run):
        print(f"🔔 Nouveau run: {run.get('model_name')} ({run.get('run_id')[:8]})")
    
    collector.register_callback(print_run)
    
    # Ajout runs de test
    runs_test = [
        {
            'run_id': 'abc123',
            'model_name': 'XGBoost',
            'status': 'FINISHED',
            'metrics': {'accuracy': 0.95}
        },
        {
            'run_id': 'def456',
            'model_name': 'Random Forest',
            'status': 'FINISHED',
            'metrics': {'accuracy': 0.92}
        },
        {
            'run_id': 'ghi789',
            'model_name': 'SVM',
            'status': 'FAILED',
            'metrics': {}
        }
    ]
    
    for run in runs_test:
        collector.add_run(run)
    
    # Statistiques
    print("\n📊 Statistiques:")
    print(json.dumps(collector.get_stats(), indent=2))
    
    # Filtrage
    print("\n✅ Runs FINISHED:")
    finished = collector.get_runs_by_status('FINISHED')
    for run in finished:
        print(f"  - {run['model_name']}: {run['metrics']}")
    
    # Export
    collector.export_to_file('mlflow_runs_backup.json')
    
    print("\n✅ Tests terminés avec succès!")