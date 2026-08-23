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

import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls", ".json"}

# Correctif Phase 1 (AUDIT_BACKEND_2026-08-23.md, Axe C.2) — `.xlsx` est un
# conteneur ZIP : sans cette limite, un classeur de quelques Mo conçu pour
# décompresser en plusieurs Go (bombe zip) pouvait faire exploser la
# mémoire d'un worker gunicorn — l'upload est synchrone dans la requête
# HTTP, pas une tâche de fond. Généreux (aligné sur le plafond déjà accepté
# pour un dataset vision, `max_vision_upload_size_mb`) : vise à bloquer une
# bombe, pas un tableur légitime volumineux.
MAX_XLSX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024

# Signatures binaires réelles (Phase 1, AUDIT_BACKEND_2026-08-23.md §C.2) —
# avant ce correctif, seule l'extension déclarée par le nom de fichier était
# vérifiée (`validate_extension`) ; un fichier renommé était accepté sans
# broncher jusqu'au parsing pandas. Ne remplace pas `validate_extension`
# (toujours utile pour choisir le bon parseur), la complète.
_ZIP_MAGIC = b"PK\x03\x04"
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_PARQUET_MAGIC = b"PAR1"


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


def _check_xlsx_not_a_bomb(path: Path) -> None:
    """Somme les tailles décompressées déclarées dans le répertoire central
    du ZIP — lecture des métadonnées seulement, jamais le contenu — AVANT
    de laisser `openpyxl` (via `pandas.read_excel`) décompresser quoi que ce
    soit. Même principe que `services/vision_datasets.py::_accumulate_member`
    pour les archives vision."""
    try:
        with zipfile.ZipFile(path) as zf:
            total_uncompressed = sum(info.file_size for info in zf.infolist())
    except zipfile.BadZipFile as exc:
        raise DatasetParsingError("Fichier .xlsx invalide (archive ZIP corrompue ou illisible).") from exc
    if total_uncompressed > MAX_XLSX_UNCOMPRESSED_BYTES:
        raise DatasetParsingError(
            "Classeur Excel refusé par précaution : trop volumineux une fois décompressé "
            f"(> {MAX_XLSX_UNCOMPRESSED_BYTES // (1024 * 1024)} Mo, bombe zip potentielle)."
        )


def _validate_signature(path: Path, extension: str) -> None:
    """Vérifie le contenu réel du fichier, pas seulement l'extension
    déclarée par son nom (Phase 1, AUDIT_BACKEND_2026-08-23.md §C.2)."""
    with open(path, "rb") as f:
        header = f.read(8)
    if extension == ".xlsx":
        if not header.startswith(_ZIP_MAGIC):
            raise DatasetParsingError("Le contenu du fichier ne correspond pas à un classeur Excel (.xlsx) valide.")
        _check_xlsx_not_a_bomb(path)
    elif extension == ".xls":
        if not header.startswith(_OLE_MAGIC):
            raise DatasetParsingError("Le contenu du fichier ne correspond pas à un classeur Excel (.xls) valide.")
    elif extension == ".parquet":
        footer = b""
        try:
            with open(path, "rb") as f:
                f.seek(-4, 2)
                footer = f.read(4)
        except OSError:
            pass  # fichier trop court — le contrôle d'en-tête ci-dessous suffit à le rejeter
        if not header.startswith(_PARQUET_MAGIC) or footer != _PARQUET_MAGIC:
            raise DatasetParsingError("Le contenu du fichier ne correspond pas à un fichier Parquet valide.")
    elif extension in (".csv", ".json"):
        # Défense en profondeur : un .xlsx/.xls renommé en .csv échapperait
        # sinon à toute vérification de signature (CSV/JSON sont du texte
        # libre, sans magic bytes propres à détecter positivement).
        if header.startswith(_ZIP_MAGIC) or header.startswith(_OLE_MAGIC):
            raise DatasetParsingError(
                "Le contenu du fichier ne correspond pas à l'extension déclarée "
                "(fichier binaire détecté pour un format texte attendu)."
            )


def read_dataframe(path: Path, extension: str) -> pd.DataFrame:
    """Charge un fichier tabulaire en DataFrame — un seul point d'entrée par format."""
    try:
        _validate_signature(path, extension)
        if extension == ".csv":
            return pd.read_csv(path)
        if extension == ".parquet":
            return pd.read_parquet(path)
        if extension in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if extension == ".json":
            return pd.read_json(path)
    except (UnsupportedFileType, DatasetParsingError):
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
