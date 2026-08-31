"""Export de déploiement autonome pour le clustering — même principe que
`domains/training/services/deployment_export.py` (pilier supervisé), retour
utilisateur direct : "tous les modèles... doivent pouvoir être déployés dans
des plateformes par la suite". Génère un script Python qui recharge le
bundle joblib déjà exportable (`GET .../model/export`) et assigne un cluster
à de nouvelles observations, SANS jamais importer un module `domains.*` de
ce projet.

Reproduit fidèlement les 3 méthodes d'assignation de
`services/inference.py::assign_cluster`/`assign_clusters_batch` (voir ce
module pour le raisonnement complet de chacune) — jamais une approximation
supplémentaire inventée pour l'occasion, la logique embarquée est une copie
exacte, prouvée par un test en sous-processus réel (comme pour le pilier
supervisé)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from string import Template
from typing import Any

_SCRIPT_TEMPLATE = Template('''#!/usr/bin/env python3
"""Script de déploiement autonome — clustering — généré par DataLab Pro le
$generated_at.

Algorithme : $algorithm ($family)
Variables attendues ($n_features) : $feature_columns_list
Méthode d'assignation pour ce modèle : $assignment_method_label

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
($artifact_filename) contient déjà le préprocesseur entraîné et le modèle de
clustering — ce script reproduit uniquement la fine couche d'orchestration
(construction du tableau d'entrée, assignation au cluster le plus proche).
$assignment_method_note"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_PATH = Path(__file__).parent / "$artifact_filename"
FEATURE_COLUMNS = $feature_columns_repr
ALGORITHM_ID = $algorithm_id_repr
_EXACT_PREDICT_ALGORITHM_IDS = {"kmeans", "minibatch_kmeans"}


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


def assign(rows: list[dict]) -> pd.DataFrame:
    """Assigne un cluster à une liste d'observations (dict colonne -> valeur
    brute). Retourne un DataFrame avec les colonnes d'origine + `cluster_id`
    (vide si non assignable), `is_noise` et `assignment_method` — jamais un
    chiffre affiché sans dire comment il a été obtenu (même principe que la
    plateforme, voir `services/inference.py`)."""
    bundle = joblib.load(ARTIFACT_PATH)
    df = _build_frame(rows)
    X_proc = bundle["preprocessor"].transform(df)
    X_proc = np.asarray(X_proc.todense()) if hasattr(X_proc, "todense") else np.asarray(X_proc)

    result = pd.DataFrame(rows)
    result["cluster_id"] = pd.array([None] * len(result), dtype="Int64")
    result["is_noise"] = False
    result["assignment_method"] = "unsupported"

    model = bundle.get("model")
    if ALGORITHM_ID in _EXACT_PREDICT_ALGORITHM_IDS and model is not None:
        result["cluster_id"] = model.predict(X_proc).astype("int64")
        result["assignment_method"] = "exact"
        return result

    if ALGORITHM_ID == "hierarchical":
        centroids = bundle.get("centroids")
        if not centroids:
            return result
        cluster_ids_sorted = sorted(centroids)
        centroid_matrix = np.stack([centroids[cid] for cid in cluster_ids_sorted])
        distances = np.linalg.norm(X_proc[:, None, :] - centroid_matrix[None, :, :], axis=2)
        nearest_idx = distances.argmin(axis=1)
        result["cluster_id"] = np.asarray(cluster_ids_sorted)[nearest_idx].astype("int64")
        result["assignment_method"] = "approximate_centroid"
        return result

    if ALGORITHM_ID == "dbscan":
        core_points = bundle.get("core_points")
        core_labels = bundle.get("core_labels")
        eps = bundle.get("eps")
        if core_points is None or eps is None or len(core_points) == 0:
            return result
        core_points = np.asarray(core_points)
        distances = np.linalg.norm(X_proc[:, None, :] - core_points[None, :, :], axis=2)
        nearest_idx = distances.argmin(axis=1)
        nearest_dist = distances[np.arange(len(distances)), nearest_idx]
        is_core_reachable = nearest_dist <= eps
        result["assignment_method"] = "approximate_nearest_core"
        result.loc[is_core_reachable, "cluster_id"] = np.asarray(core_labels)[nearest_idx][is_core_reachable].astype(
            "int64"
        )
        result.loc[~is_core_reachable, "is_noise"] = True
        return result

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignation de cluster hors ligne (modèle exporté de DataLab Pro).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", metavar="JSON", help="Une observation, au format JSON.")
    group.add_argument("--batch", nargs=2, metavar=("ENTREE.csv", "SORTIE.csv"), help="Assignation sur un CSV entier.")
    args = parser.parse_args()

    if args.predict:
        row = json.loads(args.predict)
        result = assign([row])
        print(result.to_json(orient="records", indent=2, force_ascii=False))
    else:
        input_path, output_path = args.batch
        df = pd.read_csv(input_path)
        result = assign(df.to_dict(orient="records"))
        result.to_csv(output_path, index=False)
        print(f"{len(result)} ligne(s) assignée(s) -> {output_path}")


if __name__ == "__main__":
    main()
''')

_ASSIGNMENT_METHOD_LABELS: dict[str, str] = {
    "kmeans": "exacte (distance au centroïde le plus proche, identique à l'entraînement)",
    "minibatch_kmeans": "exacte (distance au centroïde le plus proche, identique à l'entraînement)",
    "hierarchical": "approximative — centroïde le plus proche (ce modèle n'a pas de règle d'assignation native, "
    "voir la fiche modèle)",
    "dbscan": "approximative — point cœur le plus proche, dans un rayon eps (ce modèle n'a pas de règle "
    "d'assignation native, voir la fiche modèle)",
}

_ASSIGNMENT_METHOD_NOTE = '''
Note — assignation approximative
    Ce modèle ({family}) n'expose pas de méthode d'assignation native pour de
    nouvelles observations (contrairement à KMeans) — l'assignation reproduite
    ici est une approximation standard de la littérature, IDENTIQUE à celle
    utilisée par DataLab Pro (voir la colonne `assignment_method` du résultat,
    jamais un chiffre affiché sans préciser comment il a été obtenu).
'''


def generate_clustering_deployment_script(
    bundle: dict[str, Any],
    feature_columns: list[str],
    algorithm: str,
    family: str,
    artifact_filename: str,
    script_filename: str,
) -> str:
    """Construit le script `.py` autonome — voir docstring du module."""
    algorithm_id = bundle.get("algorithm_id", "")
    return _SCRIPT_TEMPLATE.substitute(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        algorithm=algorithm,
        family=family,
        n_features=len(feature_columns),
        feature_columns_list=", ".join(feature_columns),
        artifact_filename=artifact_filename,
        script_filename=script_filename,
        assignment_method_label=_ASSIGNMENT_METHOD_LABELS.get(algorithm_id, "non prise en charge pour ce modèle"),
        assignment_method_note=_ASSIGNMENT_METHOD_NOTE.format(family=family)
        if algorithm_id in ("hierarchical", "dbscan")
        else "",
        feature_columns_repr=json.dumps(feature_columns, ensure_ascii=False),
        algorithm_id_repr=json.dumps(algorithm_id),
    )
