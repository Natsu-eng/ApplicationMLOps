"""Export de déploiement autonome (retour utilisateur direct : "tous les
modèles ML — supervisé, non supervisé, vision — doivent pouvoir être
déployés dans d'autres plateformes par la suite") — génère un script Python
qui recharge le bundle déjà exportable (`GET .../model/export`) et prédit,
SANS jamais importer un module `domains.*` de ce projet : l'utilisateur doit
pouvoir emporter les deux fichiers (bundle `.joblib` + ce script) hors de
DataLab Pro et les faire tourner sur n'importe quelle machine disposant des
bonnes bibliothèques.

Le bundle lui-même (préprocesseur scikit-learn + modèle) est déjà autonome
une fois désérialisé (`joblib.load`) — ce script ne fait que reproduire la
fine couche d'orchestration (`services/inference.py::predict_one`/
`predict_batch`), jamais réimporter ce module (qui dépend indirectement de
FastAPI/SQLAlchemy via ses propres imports)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from string import Template
from typing import Any

# Bibliothèque tierce nécessaire pour DÉSÉRIALISER le modèle (au-delà de
# scikit-learn, déjà nécessaire dans tous les cas pour le préprocesseur) —
# détectée depuis le VRAI module Python de l'objet, jamais depuis le libellé
# humain affiché à l'utilisateur (`MLModel.algorithm`, ex. "Forêt aléatoire
# (Random Forest)") qui ne dit rien du paquet pip réellement requis.
_LIBRARY_BY_MODULE_PREFIX: dict[str, tuple[str, str]] = {
    "lightgbm": ("LightGBM", "lightgbm"),
    "xgboost": ("XGBoost", "xgboost"),
    "catboost": ("CatBoost", "catboost"),
}


def detect_model_library(model: Any) -> tuple[str, str]:
    """(nom affichable, nom du paquet pip) de la bibliothèque nécessaire
    pour désérialiser CE modèle précis — scikit-learn seul par défaut
    (Random Forest, Extra Trees, régression linéaire/logistique, SVM, KNN,
    Naive Bayes sont tous des estimateurs scikit-learn natifs, aucun paquet
    supplémentaire à installer pour eux)."""
    module = type(model).__module__
    for prefix, info in _LIBRARY_BY_MODULE_PREFIX.items():
        if module.startswith(prefix):
            return info
    return ("scikit-learn", "scikit-learn")


_CQR_HELPER_FUNCTION = '''

def _cqr_interval(cqr: dict, X_proc_cqr: np.ndarray, point_predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intervalle de confiance conforme (CQR, calibré à l'entraînement) —
    reproduction exacte de la formule utilisée par DataLab Pro
    (services/inference.py::_cqr_interval_batch), vectorisée sur tout le lot."""
    lo = cqr["q_lo"].predict(X_proc_cqr)
    hi = cqr["q_hi"].predict(X_proc_cqr)
    center = (lo + hi) / 2
    bounds = np.array(cqr["strata_bounds"])
    qhat = np.array(cqr["qhat_per_stratum"])
    stratum = np.clip(np.searchsorted(bounds, center, side="right") - 1, 0, len(qhat) - 1)
    lo_final = lo - qhat[stratum]
    hi_final = hi + qhat[stratum]
    if cqr.get("clip_negative"):
        lo_final = np.maximum(lo_final, 0.0)
    lo_final = np.minimum(lo_final, point_predictions)
    hi_final = np.maximum(hi_final, point_predictions)
    return lo_final, hi_final
'''

_CQR_PREDICT_BLOCK = '''        if bundle.get("cqr"):
            cqr_preprocessor = bundle["cqr"]["preprocessor"]
            X_proc_cqr = cqr_preprocessor.transform(df)
            X_proc_cqr = np.asarray(X_proc_cqr.todense()) if hasattr(X_proc_cqr, "todense") else np.asarray(X_proc_cqr)
            lo, hi = _cqr_interval(bundle["cqr"], X_proc_cqr, point)
            result["intervalle_bas"] = lo
            result["intervalle_haut"] = hi
'''

_FEATURE_ENGINEERING_WARNING = '''
ATTENTION — ingénierie de variables amont détectée
    Ce modèle a été entraîné avec des variables dérivées calculées par
    DataLab Pro (ex. décomposition de date, ratios — voir l'onglet
    "Ingénierie de variables" de la fiche modèle) AVANT le préprocesseur
    inclus dans ce bundle. Cette étape n'est PAS reproduite dans ce script
    autonome : les colonnes listées ci-dessus doivent déjà être dans leur
    forme finale (déjà dérivées) au moment de l'appel. Pour un déploiement
    fidèle incluant cette étape, utilisez l'API DataLab Pro directement
    plutôt que ce script, ou reproduisez manuellement la spec exportée dans
    la fiche modèle (section "ingenierie_variables").
'''

_SCRIPT_TEMPLATE = Template('''#!/usr/bin/env python3
"""Script de déploiement autonome — généré par DataLab Pro le $generated_at.

Modèle : $algorithm ($task_type_label)
Cible entraînée : $target_column
Variables attendues ($n_features) : $feature_columns_list

INSTALLATION
    pip install $pip_requirements

UTILISATION
    1) Placez ce script à côté du fichier modèle exporté ($artifact_filename).
    2) Une seule observation :
         python $script_filename --predict '{"exemple_variable": 1.0}'
    3) Un fichier entier (CSV avec les mêmes colonnes que ci-dessus) :
         python $script_filename --batch entree.csv sortie.csv

Ce script ne dépend d'AUCUN module de la plateforme DataLab Pro — seules les
bibliothèques listées ci-dessus sont nécessaires. Le fichier modèle
($artifact_filename) contient déjà le préprocesseur entraîné (imputation,
encodage, mise à l'échelle) et le modèle — ce script reproduit uniquement la
fine couche d'orchestration (construction du tableau d'entrée, appel au
modèle$interval_doc).
$feature_engineering_warning"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_PATH = Path(__file__).parent / "$artifact_filename"
FEATURE_COLUMNS = $feature_columns_repr
TASK_TYPE = $task_type_repr
CLASS_NAMES = $class_names_repr


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
$cqr_function

def predict(rows: list[dict]) -> pd.DataFrame:
    """Prédit sur une liste d'observations (dict colonne -> valeur brute).
    Retourne un DataFrame avec les colonnes d'origine + les colonnes de
    prédiction (jamais un sous-ensemble — permet de recroiser le résultat
    avec vos propres colonnes, ex. un identifiant de ligne)."""
    bundle = joblib.load(ARTIFACT_PATH)
    df = _build_frame(rows)
    X_proc = bundle["preprocessor"].transform(df)
    X_proc = np.asarray(X_proc.todense()) if hasattr(X_proc, "todense") else np.asarray(X_proc)

    model = bundle["model"]
    result = pd.DataFrame(rows)

    if TASK_TYPE == "classification":
        pred_indices = model.predict(X_proc)
        result["prediction"] = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in pred_indices]
        if hasattr(model, "predict_proba"):
            result["confiance"] = model.predict_proba(X_proc).max(axis=1)
    else:
        point = model.predict(X_proc)
        result["prediction"] = point
$cqr_predict_block    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prédiction hors ligne avec un modèle exporté de DataLab Pro.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", metavar="JSON", help="Une observation, au format JSON.")
    group.add_argument("--batch", nargs=2, metavar=("ENTREE.csv", "SORTIE.csv"), help="Prédiction sur un CSV entier.")
    args = parser.parse_args()

    if args.predict:
        row = json.loads(args.predict)
        result = predict([row])
        print(result.to_json(orient="records", indent=2, force_ascii=False))
    else:
        input_path, output_path = args.batch
        df = pd.read_csv(input_path)
        result = predict(df.to_dict(orient="records"))
        result.to_csv(output_path, index=False)
        print(f"{len(result)} ligne(s) prédite(s) -> {output_path}")


if __name__ == "__main__":
    main()
''')


def generate_deployment_script(
    bundle: dict[str, Any],
    feature_columns: list[str],
    algorithm: str,
    task_type: str,
    target_column: str,
    artifact_filename: str,
    script_filename: str,
) -> str:
    """Construit le script `.py` autonome — voir docstring du module. Jamais
    de f-string/`.format()` direct sur un template aussi long (parenthèses
    `{}` du code Python généré lui-même en conflit garanti) : `string.
    Template` (`$variable`) sépare proprement les deux, aucun échappement
    de portions de code à la main."""
    has_cqr = bool(bundle.get("cqr"))
    library_name, pip_name = detect_model_library(bundle["model"])
    pip_requirements = "pandas numpy scikit-learn joblib" + (f" {pip_name}" if pip_name != "scikit-learn" else "")

    return _SCRIPT_TEMPLATE.substitute(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        algorithm=algorithm,
        task_type_label="classification" if task_type == "classification" else "régression",
        target_column=target_column,
        n_features=len(feature_columns),
        feature_columns_list=", ".join(feature_columns),
        pip_requirements=pip_requirements,
        artifact_filename=artifact_filename,
        script_filename=script_filename,
        interval_doc=", l'intervalle de confiance conforme" if has_cqr else "",
        feature_engineering_warning=_FEATURE_ENGINEERING_WARNING if bundle.get("feature_engineering_spec") else "",
        feature_columns_repr=json.dumps(feature_columns, ensure_ascii=False),
        task_type_repr=json.dumps(task_type),
        class_names_repr=json.dumps(bundle.get("class_names") or []),
        cqr_function=_CQR_HELPER_FUNCTION if has_cqr else "",
        cqr_predict_block=_CQR_PREDICT_BLOCK if has_cqr else "",
    )
