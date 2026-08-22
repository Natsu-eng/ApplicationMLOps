"""Aperçu d'un dataset tabulaire déjà chargé en DataFrame — schéma +
échantillon de lignes pour la réponse d'upload/detail. Séparé de
`domains/shared/dataset_io.py` (Lot 8) : ces deux fonctions ne sont
jamais consommées hors du router `datasets`, contrairement à
`read_dataset_dataframe`/`DatasetParsingError`, génériques aux domaines ML."""
from __future__ import annotations

from typing import Any

import pandas as pd


def extract_schema(df: pd.DataFrame) -> list[dict[str, str]]:
    """Nom + type (string) de chaque colonne — sérialisable en JSON."""
    return [{"name": str(col), "dtype": str(dtype)} for col, dtype in df.dtypes.items()]


def sample_rows(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    """Échantillon JSON-safe (NaN → None) des premières lignes, pour l'aperçu."""
    sample = df.head(limit)
    return sample.where(pd.notnull(sample), None).to_dict(orient="records")
