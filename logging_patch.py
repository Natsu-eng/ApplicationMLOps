"""
PATCH URGENT - Correction globale du logging
"""

import logging
import sys

class SafeLogger:
    """Wrapper de logger qui filtre les kwargs problématiques"""
    
    def __init__(self, original_logger):
        self.logger = original_logger
        # Liste des kwargs supportés par le logging standard
        self.supported_args = {'exc_info', 'stack_info', 'stacklevel', 'extra'}
    
    def _safe_log(self, level, msg, *args, **kwargs):
        """Méthode de logging sécurisée"""
        try:
            # Filtrer uniquement les kwargs supportés
            safe_kwargs = {k: v for k, v in kwargs.items() if k in self.supported_args}
            
            # Si on a des kwargs non supportés, les formater dans le message
            unsupported_kwargs = {k: v for k, v in kwargs.items() if k not in self.supported_args}
            
            if unsupported_kwargs:
                # Formater le message avec les kwargs non supportés
                formatted_msg = f"{msg} - " + " - ".join(
                    f"{k}: {v}" for k, v in unsupported_kwargs.items()
                )
                self.logger.log(level, formatted_msg, *args, **safe_kwargs)
            else:
                self.logger.log(level, msg, *args, **safe_kwargs)
                
        except Exception as e:
            # Fallback ultra-sécurisé
            try:
                self.logger.log(level, f"SAFE_LOG_ERROR: {msg}", *args, **{'exc_info': True})
            except:
                print(f"FALLBACK LOG: {msg}")
    
    def info(self, msg, *args, **kwargs):
        self._safe_log(logging.INFO, msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._safe_log(logging.ERROR, msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self._safe_log(logging.WARNING, msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self._safe_log(logging.DEBUG, msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        kwargs['exc_info'] = True
        self._safe_log(logging.ERROR, msg, *args, **kwargs)
    
    # Déléguer les autres attributs au logger original
    def __getattr__(self, name):
        return getattr(self.logger, name)

# Application du patch global
def apply_logging_patch():
    """Applique le patch de logging sécurisé"""
    try:
        import src.shared.logging as shared_logging
        
        # Sauvegarder la fonction originale
        original_get_logger = shared_logging.get_logger
        
        # Remplacer par notre version sécurisée
        def patched_get_logger(name):
            original_logger = original_get_logger(name)
            return SafeLogger(original_logger)
        
        shared_logging.get_logger = patched_get_logger
        print("✅ Patch de logging appliqué avec succès")
        
    except Exception as e:
        print(f"⚠️ Impossible d'appliquer le patch complet: {e}")
        # Fallback: patcher le logging standard
        original_getLogger = logging.getLogger
        
        def patched_getLogger(name):
            original_logger = original_getLogger(name)
            return SafeLogger(original_logger)
        
        logging.getLogger = patched_getLogger
        print("✅ Patch de fallback appliqué")

# Appliquer automatiquement
apply_logging_patch()