"""Logique pure de traitement des datasets tabulaires — sans dépendance HTTP
ni Streamlit, directement testable. Portage simplifié de
`src/data/data_loader.py` (app Streamlit historique, voir diagnostic de
migration section C) : mêmes formats supportés, sans le couplage à
`st.session_state`/`st.cache_data` identifié dans le diagnostic.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

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


def extract_schema(df: pd.DataFrame) -> list[dict[str, str]]:
    """Nom + type (string) de chaque colonne — sérialisable en JSON."""
    return [{"name": str(col), "dtype": str(dtype)} for col, dtype in df.dtypes.items()]


def sample_rows(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    """Échantillon JSON-safe (NaN → None) des premières lignes, pour l'aperçu."""
    sample = df.head(limit)
    return sample.where(pd.notnull(sample), None).to_dict(orient="records")
