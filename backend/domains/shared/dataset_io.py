"""Lecture d'un dataset tabulaire depuis le disque — extrait de
`domains/datasets/services/dataset_io.py` (Lot 8, correctif de frontières) :
`read_dataset_dataframe`/`DatasetParsingError` sont consommés à l'identique
par TOUS les domaines ML tabulaires (training, clustering, dimensionality,
anomalies), pas seulement par le domaine `datasets` — repository pattern
générique (Lot 8, DECISIONS.md D8.1), pas une logique métier propre au
domaine `datasets`. Seuls `extract_schema`/`sample_rows` (aperçu d'upload,
jamais consommés hors du router `datasets`) restent domaine-locaux.

Portage simplifié de `src/data/data_loader.py` (app Streamlit historique,
voir diagnostic de migration section C) : mêmes formats supportés, sans le
couplage à `st.session_state`/`st.cache_data` identifié dans le diagnostic.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls", ".json"}


class UnsupportedFileType(ValueError):
    """Extension de fichier non supportée."""


class DatasetParsingError(ValueError):
    """Le fichier a été reçu mais n'a pas pu être interprété comme un tableau."""


def validate_extension(filename: str) -> str:
    """Retourne l'extension (minuscules) si supportée, lève sinon."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"Format non supporté ({extension or 'sans extension'}). "
            f"Formats acceptés : {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return extension


def read_dataframe(path: Path, extension: str) -> pd.DataFrame:
    """Charge un fichier tabulaire en DataFrame — un seul point d'entrée par format."""
    try:
        if extension == ".csv":
            return pd.read_csv(path)
        if extension == ".parquet":
            return pd.read_parquet(path)
        if extension in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if extension == ".json":
            return pd.read_json(path)
    except UnsupportedFileType:
        raise
    except Exception as exc:  # pandas lève des types d'erreur variés selon le format
        raise DatasetParsingError(f"Impossible de lire le fichier : {exc}") from exc
    raise UnsupportedFileType(extension)


@lru_cache(maxsize=64)
def _read_cached(path_str: str, extension: str, mtime_ns: int) -> pd.DataFrame:
    return read_dataframe(Path(path_str), extension)


def read_dataset_dataframe(path: Path, extension: str) -> pd.DataFrame:
    """Comme `read_dataframe`, mais met en cache (LRU) la lecture d'un
    dataset déjà persisté — un fichier de dataset ne change jamais après
    l'upload (voir `get_dataset_eda`), donc les multiples endpoints EDA
    (preview/eda/histogram/quality-check/...) appelés depuis une même page
    partagent une seule lecture disque au lieu d'une par requête (Lot 4,
    I4). `mtime_ns` dans la clé de cache : défense en profondeur si un
    fichier était un jour réécrit sur place. Réservé aux lectures d'un
    dataset déjà en base — l'upload (aucun id encore attribué) continue
    d'appeler `read_dataframe` directement.

    Retourne une copie : aucun appelant ne doit pouvoir muter l'entrée
    partagée du cache.
    """
    mtime_ns = path.stat().st_mtime_ns
    return _read_cached(str(path), extension, mtime_ns).copy()
