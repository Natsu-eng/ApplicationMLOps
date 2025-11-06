"""
Décorateurs de monitoring et gestion d'erreurs pour l'application.
Fournit des outils pour monitorer les performances des fonctions critiques,
gérer les erreurs de manière robuste, et assurer la stabilité des opérations ML.
"""
import concurrent
import time
import functools
from typing import Any, Callable, Optional

# Import centralisé du système de logging
from src.shared.logging import get_logger

logger = get_logger(__name__)

# Variable globale pour la disponibilité de psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil non disponible, monitoring mémoire limité")


def monitor_performance(operation_name: Optional[str] = None) -> Callable:
    """
    Décorateur pour monitorer les performances des fonctions critiques.
    """
    def actual_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not PSUTIL_AVAILABLE:
                return func(*args, **kwargs)
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                duration = end_time - start_time
                memory_delta = end_memory - start_memory
                
                func_name = operation_name if operation_name else func.__name__
                
                # Logs détaillés
                logger.debug(f"{func_name} - Duration: {duration:.2f}s, Memory: {memory_delta:+.1f}MB")
                
                if duration > 30:
                    logger.warning(f"⏰ {func_name} took {duration:.2f}s - performance issue")
                if memory_delta > 500:
                    logger.warning(f"💾 {func_name} used {memory_delta:.1f}MB - memory issue")
                
                return result
                
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    logger.error(f"❌ {func.__name__} received unexpected kwargs: {list(kwargs.keys())}")
                raise
            except Exception as e:
                func_name = operation_name if operation_name else func.__name__
                logger.error(f"❌ Error in {func_name}: {str(e)}", exc_info=True)
                raise
        
        return wrapper
    
    if callable(operation_name):
        return actual_decorator(operation_name)
    return actual_decorator

def monitor_ml_operation(func: Callable) -> Callable:
    """Décorateur de monitoring pour les opérations ML"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2) if PSUTIL_AVAILABLE else 0
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            memory_delta = (psutil.virtual_memory().used / (1024**2) - start_memory) if PSUTIL_AVAILABLE else 0
            
            # Seuils configurables
            if elapsed > 30:  # 30 secondes
                logger.warning(f"Opération lente: {func.__name__} a pris {elapsed:.2f}s")
            if memory_delta > 500:  # 500 MB
                logger.warning(f"Usage mémoire élevé: {func.__name__} a utilisé {memory_delta:.1f}MB")
                
            return result
        except Exception as e:
            logger.error(f"Échec {func.__name__}: {str(e)[:100]}")
            raise
    return wrapper

def safe_execute(fallback_value: Any = None, log_errors: bool = True, max_retries: int = 0) -> Callable:
    """
    Décorateur pour l'exécution sécurisée avec fallback et support retry.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        if log_errors:
                            # LOGGING STANDARD
                            logger.warning(
                                f"⚠️ Tentative {attempt + 1}/{max_retries + 1} échouée pour {func.__name__}: "
                                f"{str(e)[:100]} - Retry..."
                            )
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    else:
                        if log_errors:
                            # LOGGING STANDARD
                            logger.error(
                                f"❌ Échec définitif de {func.__name__} après {max_retries + 1} tentatives: "
                                f"{str(last_exception)}",
                                exc_info=True
                            )
            
            return fallback_value
        return wrapper
    
    return decorator

def timeout(seconds: int = 300) -> Callable:
    """Décorateur de timeout pour les opérations longues."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(func, *args, **kwargs)
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    logger.error(f"⏰ Timeout: {func.__name__} > {seconds}s")
                    return None
                except Exception as e:
                    logger.error(f"❌ Exception in {func.__name__}: {str(e)}")
                    return None
        return wrapper
    return decorator

def safe_metric_calculation(fallback_value: Any = None, max_retries: int = 1) -> Callable:
    """Décorateur robuste pour calculs de métriques avec retry."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"⚠️ Tentative {attempt + 1} échouée pour {func.__name__}, retry...")
                        time.sleep(0.1)  # Backoff minimal
                    else:
                        logger.error(f"❌ Échec après {max_retries + 1} tentatives pour {func.__name__}")
            
            return fallback_value
        return wrapper
    return decorator

def monitor_operation(func: Callable) -> Callable:
    """
    Décorateur pour monitorer les opérations ML avec logs clairs et lisibles.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        operation_name = func.__name__
        start_time = time.time()
        
        # Log de démarrage - STANDARD
        logger.info(f"🔄 Démarrage: {operation_name}")
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log de succès - STANDARD
            logger.info(
                f"✅ Succès: {operation_name} | "
                f"Durée: {duration:.2f}s | "
                f"Module: {func.__module__}"
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)[:200]
            
            # Log d'erreur - STANDARD
            logger.error(
                f"❌ Échec: {operation_name} | "
                f"Durée: {duration:.2f}s | "
                f"Erreur: {error_msg}"
            )
            
            raise
    
    return wrapper

def handle_mlflow_errors(func: Callable) -> Callable:
    """
    Décorateur pour gestion spécifique des erreurs MLflow.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except ImportError:
            logger.warning("MLflow non disponible - poursuite sans tracking")
            return None
        except Exception as e:
            logger.error(
                f"Erreur MLflow dans {func.__name__}: {str(e)[:100]}",
                exc_info=True
            )
            return None
    
    return wrapper