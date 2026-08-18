"""Tests de services/datasets.py::read_dataset_dataframe (Lot 4, correctif
I4, AUDIT_DATALAB_2026-08-16.md §I4) — le cache LRU en mémoire qui évite de
relire le fichier à chaque appel EDA/preview/histogram/job sur un même
dataset. Fichiers réels sur disque (tmp_path), aucune base ni API."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from services.datasets import DatasetParsingError, _read_cached, read_dataset_dataframe


@pytest.fixture(autouse=True)
def _clear_cache():
    _read_cached.cache_clear()
    yield
    _read_cached.cache_clear()


def _write_csv(tmp_path, name: str, rows: int) -> Path:
    path = tmp_path / name
    pd.DataFrame({"a": range(rows)}).to_csv(path, index=False)
    return path


def test_repeated_calls_hit_the_cache_not_the_disk(tmp_path):
    path = _write_csv(tmp_path, "d.csv", 3)
    read_dataset_dataframe(path, ".csv")
    read_dataset_dataframe(path, ".csv")
    info = _read_cached.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_returned_frame_is_a_copy_mutating_it_does_not_poison_the_cache(tmp_path):
    path = _write_csv(tmp_path, "d.csv", 3)
    df1 = read_dataset_dataframe(path, ".csv")
    df1["a"] = -1
    df2 = read_dataset_dataframe(path, ".csv")
    assert list(df2["a"]) == [0, 1, 2]


def test_file_rewritten_on_disk_invalidates_the_cache_via_mtime(tmp_path):
    path = _write_csv(tmp_path, "d.csv", 3)
    df1 = read_dataset_dataframe(path, ".csv")
    assert len(df1) == 3

    pd.DataFrame({"a": range(7)}).to_csv(path, index=False)
    # Force un mtime distinct : sur certains systèmes de fichiers, deux
    # écritures rapprochées dans un même test peuvent partager la même
    # valeur de mtime (résolution de l'horloge), ce qui masquerait le
    # comportement d'invalidation qu'on veut vérifier ici.
    current = path.stat().st_mtime_ns
    os.utime(path, ns=(current + 1_000_000_000, current + 1_000_000_000))

    df2 = read_dataset_dataframe(path, ".csv")
    assert len(df2) == 7
    assert _read_cached.cache_info().currsize == 2


def test_parsing_error_is_not_cached_and_is_retried(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_bytes(b"\x00\x01\x02\x03not,a,csv\x00")
    with pytest.raises(DatasetParsingError):
        read_dataset_dataframe(path, ".xlsx")
    with pytest.raises(DatasetParsingError):
        read_dataset_dataframe(path, ".xlsx")
    # lru_cache ne mémorise jamais une exception : rien n'est entré en
    # cache, donc rien à invalider si le fichier est corrigé ensuite.
    assert _read_cached.cache_info().currsize == 0
