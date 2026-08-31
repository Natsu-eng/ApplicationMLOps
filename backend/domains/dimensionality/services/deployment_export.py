"""Export de déploiement autonome pour la réduction de dimension — même
principe que `domains/clustering/services/deployment_export.py`. Génère un
script Python qui recharge le bundle joblib déjà exportable (`GET
.../model/export`) et projette de nouvelles observations, SANS jamais
importer un module `domains.*` de ce projet.

Honnête sur les limites : PCA et UMAP supportent nativement `.transform()`
(projection EXACTE) ; t-SNE est un modèle transductif (sklearn n'expose
aucun `.transform()`) — le script généré pour un job t-SNE le rappelle
explicitement au lieu d'échouer silencieusement ou d'inventer une
approximation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from string import Template

# t-SNE (sklearn) n'a pas de `.transform()` — aucune bibliothèque tierce ne
# comble ce manque de façon fidèle sans ré-entraîner sur l'ensemble des
# données (ce qu'un script de déploiement autonome ne fait jamais).
_UNSUPPORTED_ALGORITHM_IDS = {"tsne"}

_UNSUPPORTED_NOTE = '''
ATTENTION — projection de nouvelles observations non disponible
    t-SNE est un modèle transductif : il n'existe littéralement aucune
    méthode pour situer un nouveau point dans l'embedding déjà calculé sans
    ré-entraîner sur l'ensemble des données (contrairement à PCA/UMAP). Ce
    script charge quand même le bundle et peut relire la projection déjà
    calculée pour le jeu d'entraînement, mais `project()` lèvera une erreur
    explicite sur toute NOUVELLE observation. Pour projeter de nouvelles
    données, ré-entraînez avec PCA ou UMAP dans DataLab Pro.
'''

_SCRIPT_TEMPLATE = Template('''#!/usr/bin/env python3
"""Script de déploiement autonome — réduction de dimension — généré par
DataLab Pro le $generated_at.

Méthode : $algorithm
Variables attendues ($n_features) : $feature_columns_list

INSTALLATION
    pip install pandas numpy scikit-learn joblib$extra_pip

UTILISATION
    1) Placez ce script à côté du fichier modèle exporté ($artifact_filename).
    2) Une seule observation :
         python $script_filename --predict '{"exemple_variable": 1.0}'
    3) Un fichier entier (CSV avec les mêmes colonnes que ci-dessus) :
         python $script_filename --batch entree.csv sortie.csv

Ce script ne dépend d'AUCUN module de la plateforme DataLab Pro — seules les
bibliothèques listées ci-dessus sont nécessaires. Le fichier modèle
($artifact_filename) contient déjà le préprocesseur entraîné et le modèle de
projection ($algorithm).
$unsupported_note"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_PATH = Path(__file__).parent / "$artifact_filename"
FEATURE_COLUMNS = $feature_columns_repr
SUPPORTS_NEW_POINTS = $supports_new_points_repr


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


def project(rows: list[dict]) -> pd.DataFrame:
    """Projette une liste d'observations (dict colonne -> valeur brute) sur
    les 2 axes de la réduction de dimension. Retourne un DataFrame avec les
    colonnes d'origine + `x`/`y`."""
    if not SUPPORTS_NEW_POINTS:
        raise RuntimeError(
            "Cette méthode ($algorithm) ne permet pas de projeter de nouvelles observations "
            "(modèle transductif) — voir l'en-tête de ce script."
        )
    bundle = joblib.load(ARTIFACT_PATH)
    df = _build_frame(rows)
    X_proc = bundle["preprocessor"].transform(df)
    X_proc = np.asarray(X_proc.todense()) if hasattr(X_proc, "todense") else np.asarray(X_proc)

    embedding = np.asarray(bundle["primary_model"].transform(X_proc))
    result = pd.DataFrame(rows)
    result["x"] = embedding[:, 0]
    result["y"] = embedding[:, 1] if embedding.shape[1] > 1 else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Projection hors ligne (modèle exporté de DataLab Pro).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", metavar="JSON", help="Une observation, au format JSON.")
    group.add_argument("--batch", nargs=2, metavar=("ENTREE.csv", "SORTIE.csv"), help="Projection sur un CSV entier.")
    args = parser.parse_args()

    if args.predict:
        row = json.loads(args.predict)
        result = project([row])
        print(result.to_json(orient="records", indent=2, force_ascii=False))
    else:
        input_path, output_path = args.batch
        df = pd.read_csv(input_path)
        result = project(df.to_dict(orient="records"))
        result.to_csv(output_path, index=False)
        print(f"{len(result)} ligne(s) projetée(s) -> {output_path}")


if __name__ == "__main__":
    main()
''')


def generate_dimensionality_deployment_script(
    feature_columns: list[str],
    algorithm: str,
    algorithm_id: str,
    artifact_filename: str,
    script_filename: str,
) -> str:
    """Construit le script `.py` autonome — voir docstring du module."""
    supports_new_points = algorithm_id not in _UNSUPPORTED_ALGORITHM_IDS
    extra_pip = " umap-learn" if algorithm_id == "umap" else ""
    return _SCRIPT_TEMPLATE.substitute(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        algorithm=algorithm,
        n_features=len(feature_columns),
        feature_columns_list=", ".join(feature_columns),
        extra_pip=extra_pip,
        artifact_filename=artifact_filename,
        script_filename=script_filename,
        unsupported_note=_UNSUPPORTED_NOTE if not supports_new_points else "",
        feature_columns_repr=json.dumps(feature_columns, ensure_ascii=False),
        supports_new_points_repr=repr(supports_new_points),
    )
