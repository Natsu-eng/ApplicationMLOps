import pandas as pd
import dask.dataframe as dd
import streamlit as st
from typing import Dict, Any, Tuple, Union, Optional
import os
import re
import time
import gc

# Import des modules déplacés
from monitoring.decorators import monitor_performance, safe_execute
from helpers.data_transformers import intelligent_type_coercion, optimize_dataframe
from utils.file_utils import validate_file_integrity, get_file_extension, is_supported_extension, get_file_size_mb

# Configuration du logging pour production
from src.shared.logging import get_logger
logger = get_logger(__name__)

# Extensions de fichiers supportées
SUPPORTED_EXTENSIONS = {'csv', 'parquet', 'xlsx', 'xls', 'json'}
# Taille maximale du fichier en Mo (1 Go)
MAX_FILE_SIZE_MB = 1024


@safe_execute(fallback_value=(None, {"actions": ["Erreur critique lors du chargement"]}, None))
@monitor_performance
def load_data(
    file_path: str,
    force_dtype: Dict[str, Any] = None,
    sanitize_for_display: bool = True,
    size_threshold_mb: float = 100.0,
    blocksize: str = "64MB"
) -> Tuple[Union[pd.DataFrame, dd.DataFrame], Dict[str, Any], Union[pd.DataFrame, dd.DataFrame]]:
    """
    Charge les données depuis un fichier et décide automatiquement d'utiliser Pandas ou Dask.
    Version optimisée pour la production avec validation et monitoring.
    
    Args:
        file_path: Chemin du fichier à charger (str ou file-like object)
        force_dtype: Dictionnaire des types forcés pour les colonnes
        sanitize_for_display: Si True, applique la coercion intelligente
        size_threshold_mb: Seuil en Mo pour basculer vers Dask
        blocksize: Taille des blocs pour Dask
    
    Returns:
        Tuple contenant le DataFrame, un rapport d'actions, et le DataFrame brut
    """
    report = {"actions": [], "changes": {}, "warnings": []}
    
    try:
        # Détermination du nom et extension du fichier
        if isinstance(file_path, str):
            file_name = os.path.basename(file_path)
        else:
            file_name = getattr(file_path, 'name', 'fichier_uploadé')
            
        file_extension = file_name.split('.')[-1].lower() if '.' in file_name else ''
        
        # Validation de l'extension
        if file_extension not in SUPPORTED_EXTENSIONS:
            error_msg = f"Extension de fichier non supportée : {file_extension}. Extensions valides : {', '.join(SUPPORTED_EXTENSIONS)}"
            logger.error(error_msg)
            return None, {"actions": [f"Erreur : {error_msg}"]}, None

        # Validation de la taille du fichier
        if isinstance(file_path, str):
            if not os.path.exists(file_path):
                error_msg = f"Fichier non trouvé : {file_path}"
                logger.error(error_msg)
                return None, {"actions": [f"Erreur : {error_msg}"]}, None
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        else:
            file_size_mb = getattr(file_path, 'size', 0) / (1024 * 1024)
            
        if file_size_mb > MAX_FILE_SIZE_MB:
            error_msg = f"Taille du fichier ({file_size_mb:.2f} Mo) dépasse la limite de {MAX_FILE_SIZE_MB} Mo"
            logger.error(error_msg)
            return None, {"actions": [f"Erreur : {error_msg}"]}, None

        # Validation de l'intégrité du fichier
        validation_result = validate_file_integrity(file_path, file_extension)
        if not validation_result["is_valid"]:
            error_msg = f"Fichier corrompu ou invalide : {'; '.join(validation_result['issues'])}"
            logger.error(error_msg)
            return None, {"actions": [f"Erreur : {error_msg}"]}, None
            
        # Ajouter les avertissements au rapport
        if validation_result["warnings"]:
            report["warnings"].extend(validation_result["warnings"])

        logger.info(f"Chargement du fichier : {file_name} (extension: {file_extension}, taille: {file_size_mb:.2f} Mo)")

        # Décision Pandas vs Dask
        use_dask = file_size_mb > size_threshold_mb
        logger.info(f"Utilisation de {'Dask' if use_dask else 'Pandas'} (seuil: {size_threshold_mb} Mo)")

        # Préparation des paramètres de chargement
        load_params = {}
        if force_dtype:
            load_params['dtype'] = force_dtype

        # Chargement des données selon le type de fichier et l'engine
        df = None
        
        if use_dask:
            if file_extension == 'csv':
                load_params['blocksize'] = blocksize
                df = dd.read_csv(file_path, **load_params)
                report["actions"].append(f"Chargement du fichier CSV '{file_name}' avec Dask (blocksize: {blocksize}).")
            elif file_extension == 'parquet':
                df = dd.read_parquet(file_path, **{k: v for k, v in load_params.items() if k != 'blocksize'})
                report["actions"].append(f"Chargement du fichier Parquet '{file_name}' avec Dask.")
            elif file_extension == 'json':
                df = dd.read_json(file_path, **{k: v for k, v in load_params.items() if k != 'blocksize'})
                report["actions"].append(f"Chargement du fichier JSON '{file_name}' avec Dask.")
            else:
                error_msg = f"Extension de fichier '{file_extension}' non supportée pour Dask"
                logger.error(error_msg)
                return None, {"actions": [f"Erreur : {error_msg}"]}, None
        else:
            if file_extension == 'csv':
                load_params['low_memory'] = False
                df = pd.read_csv(file_path, **load_params)
                report["actions"].append(f"Chargement du fichier CSV '{file_name}' avec Pandas (low_memory=False).")
            elif file_extension == 'parquet':
                df = pd.read_parquet(file_path)
                report["actions"].append(f"Chargement du fichier Parquet '{file_name}' avec Pandas.")
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
                report["actions"].append(f"Chargement du fichier Excel '{file_name}' avec Pandas.")
            elif file_extension == 'json':
                df = pd.read_json(file_path)
                report["actions"].append(f"Chargement du fichier JSON '{file_name}' avec Pandas.")
            else:
                error_msg = f"Extension de fichier '{file_extension}' non supportée"
                logger.error(error_msg)
                return None, {"actions": [f"Erreur : {error_msg}"]}, None

        if df is None or (not use_dask and df.empty) or (use_dask and df.npartitions == 0):
            error_msg = "Le fichier chargé est vide"
            logger.error(error_msg)
            return None, {"actions": [f"Erreur : {error_msg}"]}, None

        # Sauvegarde du DataFrame brut avant toute modification
        if use_dask:
            df_raw = df.copy()
        else:
            df_raw = df.copy()

        # Calcul du nombre de lignes initial
        initial_rows = len(df) if not use_dask else "inconnu (Dask, calcul paresseux)"

        # Suppression des doublons
        try:
            if use_dask:
                df = df.drop_duplicates()
                report["actions"].append("Suppression des lignes dupliquées (opération Dask paresseuse).")
            else:
                duplicates_count = df.duplicated().sum()
                if duplicates_count > 0:
                    df = df.drop_duplicates().reset_index(drop=True)
                    final_rows = len(df)
                    report["actions"].append(f"{duplicates_count} lignes dupliquées supprimées.")
                else:
                    report["actions"].append("Aucune ligne dupliquée détectée.")
                    
        except Exception as e:
            logger.warning(f"Erreur lors de la suppression des doublons : {e}")
            report["warnings"].append(f"Suppression des doublons échouée : {e}")

        # Coercion intelligente des types (seulement pour Pandas)
        if sanitize_for_display and not use_dask:
            try:
                df, changes = intelligent_type_coercion(df, use_dask)
                report["changes"] = changes
                if changes:
                    report["actions"].append(f"Standardisation des types : {len(changes)} colonnes modifiées.")
            except Exception as e:
                logger.error(f"Erreur lors de la coercion de types : {e}")
                report["warnings"].append(f"Coercion de types échouée : {e}")

        # Sauvegarde dans st.session_state
        try:
            st.session_state.df = df
            st.session_state.df_raw = df_raw
            logger.info("DataFrames sauvegardés dans session_state")
        except Exception as e:
            logger.warning(f"Erreur sauvegarde session_state : {e}")

        # Statistiques finales
        final_rows = len(df) if not use_dask else "inconnu (Dask)"
        final_cols = len(df.columns)
        
        logger.info(f"Données chargées avec succès : {final_rows} lignes et {final_cols} colonnes")
        report["actions"].append(f"Dataset final : {final_rows} lignes × {final_cols} colonnes")
        
        return df, report, df_raw

    except Exception as e:
        error_msg = f"Erreur critique lors du chargement du fichier : {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, {"actions": [error_msg]}, None