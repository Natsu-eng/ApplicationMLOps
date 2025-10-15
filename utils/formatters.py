"""
Fonctions de formatage pour l'application.
Étendu avec les formateurs pour les métriques.
"""
import numpy as np
from typing import Any

def format_metric_value(value: Any, precision: int = 3) -> str:
    """Formate une valeur métrique pour l'affichage."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    
    try:
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        
        if isinstance(value, (float, np.floating)):
            if abs(value) < 0.001 or abs(value) > 10000:
                return f"{value:.2e}"
            return f"{value:.{precision}f}"
        
        return str(value)
    except (ValueError, TypeError):
        return str(value)

def _sanitize_metrics_for_output(metrics: dict) -> dict:
    """Nettoie les métriques pour la sortie (supprime les objets complexes)."""
    sanitized = {}
    for key, value in metrics.items():
        if key in ['error', 'warnings', 'success']:
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, (list, dict)) and not isinstance(value, (np.ndarray)):
            # Conversion récursive pour les structures simples
            try:
                import json
                json.dumps(value)  # Test de sérialisation
                sanitized[key] = value
            except (TypeError, ValueError):
                continue  # Ignore les structures complexes
    return sanitized


import numpy as np
from typing import Any, Dict
class DataFormatter:
    """Formatage et transformation des données pour l'affichage"""
    
    @staticmethod
    def format_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Formate les métriques pour l'affichage"""
        formatted = {}
        for key, value in metrics.items():
            if isinstance(value, (int, np.integer)):
                formatted[key] = f"{value:,}"
            elif isinstance(value, (float, np.floating)):
                if abs(value) < 0.001:
                    formatted[key] = f"{value:.2e}"
                else:
                    formatted[key] = f"{value:.3f}"
            else:
                formatted[key] = str(value)
        return formatted
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 50) -> str:
        """Tronque le texte avec des points de suspension"""
        if not isinstance(text, str):
            text = str(text)
        return text[:max_length] + "..." if len(text) > max_length else text
    
    @staticmethod
    def format_memory_size(bytes_size: int) -> str:
        """Formate la taille mémoire en unités lisibles"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"