"""Export de déploiement autonome pour la détection d'anomalies — même
principe que `domains/clustering/services/deployment_export.py` et
`domains/training/services/deployment_export.py`. Génère un script Python
qui recharge le bundle joblib déjà exportable (`GET .../model/export`) et
note de nouvelles observations, SANS jamais importer un module `domains.*`
de ce projet.

Reproduit fidèlement `services/inference.py::score_anomaly`/
`score_anomalies_batch` (Isolation Forest nativement inductif, LOF via
l'instance dédiée `novelty=True`, rang percentile relatif aux scores BRUTS
d'entraînement persistés dans le bundle) — copie exacte, jamais une
approximation supplémentaire inventée pour l'occasion, prouvée par un test
en sous-processus réel."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from string import Template

_SCRIPT_TEMPLATE = Template('''#!/usr/bin/env python3
"""Script de déploiement autonome — détection d'anomalies — généré par
DataLab Pro le $generated_at.

Isolation Forest + Local Outlier Factor (consensus) — variables attendues
($n_features) : $feature_columns_list

INSTALLATION
    pip install pandas numpy scikit-learn joblib

UTILISATION
    1) Placez ce script à côté du fichier modèle exporté ($artifact_filename).
    2) Une seule observation :
         python $script_filename --predict '{"exemple_variable": 1.0}'
    3) Un fichier entier (CSV avec les mêmes colonnes que ci-dessus) :
         python $script_filename --batch entree.csv sortie.csv

Ce script ne dépend d'AUCUN module de la plateforme DataLab Pro — seules les
bibliothèques listées ci-dessus sont nécessaires. Le fichier modèle
($artifact_filename) contient déjà le préprocesseur, Isolation Forest, une
instance Local Outlier Factor dédiée aux nouvelles observations, et les
scores d'entraînement des deux algorithmes (nécessaires pour situer une
nouvelle observation au même rang percentile que le score de consensus
calculé par DataLab Pro — un percentile n'a de sens que relatif à une
distribution de référence)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_PATH = Path(__file__).parent / "$artifact_filename"
FEATURE_COLUMNS = $feature_columns_repr


def _build_frame(rows: list[dict]) -> pd.DataFrame:
    missing_columns = [c for c in FEATURE_COLUMNS if c not in rows[0]]
    if missing_columns:
        raise ValueError(f"Colonne(s) manquante(s) : {', '.join(missing_columns)}")
    df = pd.DataFrame(rows)[FEATURE_COLUMNS]
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if not converted.isna().any():
            df[col] = converted
    return df


def _agreement_label(is_if: bool, is_lof: bool) -> str:
    if is_if and is_lof:
        return "both"
    if is_if:
        return "isolation_forest_only"
    if is_lof:
        return "lof_only"
    return "none"


def score(rows: list[dict]) -> pd.DataFrame:
    """Note une liste d'observations (dict colonne -> valeur brute).
    Retourne un DataFrame avec les colonnes d'origine + les colonnes de
    score — jamais un chiffre affiché sans dire comment il a été obtenu
    (même principe que la plateforme, voir `services/inference.py`)."""
    bundle = joblib.load(ARTIFACT_PATH)
    df = _build_frame(rows)
    X_proc = bundle["preprocessor"].transform(df)
    X_proc = np.asarray(X_proc.todense()) if hasattr(X_proc, "todense") else np.asarray(X_proc)

    if_estimator = bundle["isolation_forest"]
    lof_novelty = bundle["lof_novelty"]
    scores_if_train = np.asarray(bundle["scores_if_train"])
    scores_lof_train = np.asarray(bundle["scores_lof_train"])

    scores_if = if_estimator.score_samples(X_proc)
    is_anomaly_if = if_estimator.predict(X_proc) == -1
    scores_lof = lof_novelty.score_samples(X_proc)
    is_anomaly_lof = lof_novelty.predict(X_proc) == -1

    rank_if = (scores_if_train[None, :] >= scores_if[:, None]).mean(axis=1)
    rank_lof = (scores_lof_train[None, :] >= scores_lof[:, None]).mean(axis=1)
    consensus = (rank_if + rank_lof) / 2.0
    is_anomaly_consensus = is_anomaly_if & is_anomaly_lof

    result = pd.DataFrame(rows)
    result["consensus_score"] = consensus
    result["score_isolation_forest"] = rank_if
    result["score_lof"] = rank_lof
    result["is_anomaly_isolation_forest"] = is_anomaly_if
    result["is_anomaly_lof"] = is_anomaly_lof
    result["is_anomaly_consensus"] = is_anomaly_consensus
    result["agreement"] = [
        _agreement_label(bool(a), bool(b)) for a, b in zip(is_anomaly_if, is_anomaly_lof, strict=True)
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Notation hors ligne (modèle de détection exporté de DataLab Pro).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", metavar="JSON", help="Une observation, au format JSON.")
    group.add_argument("--batch", nargs=2, metavar=("ENTREE.csv", "SORTIE.csv"), help="Notation sur un CSV entier.")
    args = parser.parse_args()

    if args.predict:
        row = json.loads(args.predict)
        result = score([row])
        print(result.to_json(orient="records", indent=2, force_ascii=False))
    else:
        input_path, output_path = args.batch
        df = pd.read_csv(input_path)
        result = score(df.to_dict(orient="records"))
        result.to_csv(output_path, index=False)
        print(f"{len(result)} ligne(s) notée(s) -> {output_path}")


if __name__ == "__main__":
    main()
''')


def generate_anomaly_deployment_script(
    feature_columns: list[str],
    artifact_filename: str,
    script_filename: str,
) -> str:
    """Construit le script `.py` autonome — voir docstring du module."""
    return _SCRIPT_TEMPLATE.substitute(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        n_features=len(feature_columns),
        feature_columns_list=", ".join(feature_columns),
        artifact_filename=artifact_filename,
        script_filename=script_filename,
        feature_columns_repr=json.dumps(feature_columns, ensure_ascii=False),
    )
