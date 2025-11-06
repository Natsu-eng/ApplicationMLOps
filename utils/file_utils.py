"""
Utilitaires pour la gestion des fichiers et validation.
"""
import os
import re
from typing import Dict, Any, Union
from src.shared.logging import get_logger

logger = get_logger(__name__)

# Extensions de fichiers supportées
SUPPORTED_EXTENSIONS = {'csv', 'parquet', 'xlsx', 'xls', 'json'}

def validate_file_integrity(file_path: Union[str, Any], file_extension: str) -> Dict[str, Any]:
    """
    Valide l'intégrité d'un fichier avant le chargement.
    
    Args:
        file_path: Chemin du fichier
        file_extension: Extension du fichier
    
    Returns:
        Dictionnaire avec le statut de validation
    """
    validation_report = {"is_valid": True, "issues": [], "warnings": []}
    
    try:
        if file_extension == 'csv':
            # Vérifier les premières lignes pour détecter les problèmes d'encoding
            try:
                if isinstance(file_path, str):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_lines = [f.readline() for _ in range(5)]
                else:
                    # Fichier uploadé via Streamlit
                    first_lines = []
                    for _ in range(5):
                        line = file_path.readline()
                        if not line:
                            break
                        first_lines.append(line.decode('utf-8') if isinstance(line, bytes) else line)
                    file_path.seek(0)  # Reset position
                    
                if not any(line.strip() for line in first_lines):
                    validation_report["issues"].append("Fichier CSV vide ou mal formaté")
                    validation_report["is_valid"] = False
                    
            except UnicodeDecodeError:
                validation_report["warnings"].append("Possible problème d'encodage UTF-8")
                
        elif file_extension == 'parquet':
            # Test de lecture rapide des métadonnées
            try:
                import pyarrow.parquet as pq
                if isinstance(file_path, str):
                    pq.read_metadata(file_path)
                else:
                    validation_report["warnings"].append("Validation Parquet limitée pour fichiers uploadés")
            except Exception as e:
                validation_report["issues"].append(f"Fichier Parquet corrompu: {e}")
                validation_report["is_valid"] = False
                
        elif file_extension in ['xlsx', 'xls']:
            # Vérification basique pour Excel
            try:
                import pandas as pd
                if isinstance(file_path, str):
                    # Test rapide de lecture des métadonnées Excel
                    pd.read_excel(file_path, nrows=0)
                else:
                    validation_report["warnings"].append("Validation Excel limitée pour fichiers uploadés")
            except Exception as e:
                validation_report["issues"].append(f"Fichier Excel corrompu: {e}")
                validation_report["is_valid"] = False
                
    except Exception as e:
        validation_report["issues"].append(f"Erreur de validation: {e}")
        validation_report["is_valid"] = False
        logger.error(f"❌ File validation error: {e}")
        
    return validation_report

def get_file_extension(filename: str) -> str:
    """
    Extrait l'extension d'un fichier de manière sécurisée.
    
    Args:
        filename: Nom du fichier
        
    Returns:
        Extension en minuscules
    """
    try:
        return filename.split('.')[-1].lower() if '.' in filename else ''
    except Exception:
        return ''

def is_supported_extension(filename: str) -> bool:
    """
    Vérifie si l'extension du fichier est supportée.
    
    Args:
        filename: Nom du fichier
        
    Returns:
        True si l'extension est supportée
    """
    extension = get_file_extension(filename)
    return extension in SUPPORTED_EXTENSIONS

def get_file_size_mb(file_path: Union[str, Any]) -> float:
    """
    Retourne la taille d'un fichier en Mo.
    
    Args:
        file_path: Chemin du fichier ou objet fichier
        
    Returns:
        Taille en Mo
    """
    try:
        if isinstance(file_path, str):
            return os.path.getsize(file_path) / (1024 * 1024)
        else:
            return getattr(file_path, 'size', 0) / (1024 * 1024)
    except Exception as e:
        logger.error(f"❌ Erreur calcul taille fichier: {e}")
        return 0.0