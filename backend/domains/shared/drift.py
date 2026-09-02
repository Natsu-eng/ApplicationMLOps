"""Détection de dérive des données (data drift) — pilier tabulaire.

Absente jusqu'ici : la promesse "aide à la décision" s'arrêtait au moment
du déploiement (verdict, seuil, fiabilité) mais ne disait jamais rien sur
ce qui se passe APRÈS, une fois le modèle réellement utilisé en
production. Identifiée en évaluant une maquette externe (dont le texte
anticipe "le plus ancien n'a pas été revérifié — contrôler sa dérive") :
notre propre tableau de bord ("Fiabilité des modèles actifs",
`domains/dashboard/router.py`) n'offrait jusqu'ici AUCUN moyen de le
faire — ce module comble ce trou.

Dérive DE DONNÉES uniquement (pas dérive de performance/concept) : on
compare la distribution des variables d'ENTRÉE réellement envoyées en
production (`Prediction.input_json`, déjà journalisées depuis le Lot 5)
à celle du dataset d'ENTRAÎNEMENT — jamais la qualité des prédictions
elles-mêmes, qui exigerait de connaître la vraie valeur cible pour
chaque prédiction (rarement disponible en production, sans quoi le
modèle n'aurait jamais eu besoin d'être entraîné). C'est la définition
standard de la "dérive des données" en MLOps (feature drift / covariate
shift), distincte de la dérive de concept.

Méthode : PSI (Population Stability Index), standard de l'industrie du
scoring de crédit, choisi plutôt qu'un test de Kolmogorov-Smirnov pour
deux raisons concrètes : (1) un seul nombre par variable avec des seuils
d'interprétation universellement admis (voir `PSI_*` ci-dessous), quand
un test KS ne renvoie qu'une p-value dont l'interprétation dépend de la
taille d'échantillon — un volume de prédictions élevé rendrait n'importe
quel écart minuscule "statistiquement significatif" sans être
opérationnellement pertinent ; (2) fonctionne identiquement sur les
variables numériques ET catégorielles (même formule, buckets différents),
quand KS ne s'applique qu'au numérique.

Référence : Siddiqi, N. (2006), "Credit Risk Scorecards: Developing and
Implementing Intelligent Credit Scoring", Wiley — seuils PSI < 0,1 (stable)
/ 0,1-0,25 (modéré) / > 0,25 (significatif), repris tels quels ci-dessous.

Logique pure (comme `dataset_eda.py`/`data_quality.py`/`verdict.py`) :
prend des `pd.Series` déjà chargées, ne touche ni DB ni fichier — testable
avec des données synthétiques."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Nombre de buckets pour une variable numérique — même ordre de grandeur
# que `DEFAULT_HISTOGRAM_BINS` de `dataset_eda.py` (10 plutôt que 20 : le
# PSI perd en robustesse avec trop de buckets sur un volume de prédictions
# de production souvent plus faible que le dataset d'entraînement).
PSI_BINS = 10

# Lissage epsilon — évite une division par zéro / un log(0) quand un bucket
# est vide dans l'une des deux populations (bucket jamais vu en production,
# ou catégorie de la référence jamais reproduite) : un écart réel, jamais
# un artefact numérique à masquer en le filtrant.
_PSI_EPSILON = 1e-4

# Seuils PSI (Siddiqi 2006, voir docstring du module) — bornes INCLUSES du
# côté "stable"/"modéré" (< 0.1 strictement stable, >= 0.25 strictement
# significatif), jamais recalculés dynamiquement : ce sont des repères
# métier admis, pas une propriété statistique du dataset.
PSI_STABLE_MAX = 0.1
PSI_MODERATE_MAX = 0.25

MAX_PREDICTIONS_FOR_DRIFT = 1000
# Borne le nombre de `Prediction` chargées par l'appelant (router) pour
# construire `current_df` — même esprit que `MAX_ROWS_FOR_STATS` de
# `dataset_eda.py` : au-delà, les prédictions les plus RÉCENTES (pas un
# échantillon aléatoire) suffisent largement à estimer un PSI stable, et
# borner évite de charger des milliers de lignes JSON pour un modèle
# utilisé intensivement depuis longtemps.

MIN_CURRENT_ROWS_FOR_DRIFT = 30
# En-dessous, le PSI devient statistiquement instable (un seul bucket
# vide/plein bascule le score) — mieux vaut le dire honnêtement plutôt que
# d'afficher un chiffre qui semblerait fiable sans l'être. Choisi comme
# `SHAP_SAMPLE_SIZE`/`CV_FOLDS_DEFAULT` ailleurs dans ce dépôt : un ordre
# de grandeur documenté, pas une valeur devinée.


def _psi_from_frequencies(expected_frac: np.ndarray, actual_frac: np.ndarray) -> float:
    """Formule PSI brute une fois les deux populations réduites au même
    jeu de buckets, en proportions déjà lissées (jamais de 0 exact)."""
    expected_frac = np.clip(expected_frac, _PSI_EPSILON, None)
    actual_frac = np.clip(actual_frac, _PSI_EPSILON, None)
    return float(np.sum((actual_frac - expected_frac) * np.log(actual_frac / expected_frac)))


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = PSI_BINS) -> float:
    """PSI d'UNE variable entre deux échantillons (référence = entraînement,
    actuel = prédictions récentes). Les bornes de buckets viennent TOUJOURS
    de la référence (jamais recalculées sur `current`) : le PSI mesure un
    écart PAR RAPPORT à la référence, pas une comparaison symétrique — deux
    distributions identiques mais avec des bornes de buckets différentes
    donneraient un PSI non nul si on les laissait dériver chacune de son
    côté.

    Numérique : buckets par quantiles de la référence (pas des bins de
    largeur fixe comme `compute_histogram` — équi-remplis par construction,
    ce qui évite un bucket de référence vide qui exploserait le PSI dès la
    moindre observation actuelle qui y tombe).

    Catégoriel : un bucket par catégorie observée dans la référence, plus
    un bucket "nouvelles catégories" pour toute valeur de `current` absente
    de la référence — une catégorie inédite en production est en soi un
    signal de dérive, jamais silencieusement ignorée."""
    ref = reference.dropna()
    cur = current.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    if pd.api.types.is_numeric_dtype(ref):
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(ref.astype(float).quantile(quantiles).to_numpy())
        if len(edges) < 2:
            return 0.0  # référence constante — aucune distribution à comparer
        edges[0], edges[-1] = -np.inf, np.inf  # capture toute valeur actuelle hors plage vue à l'entraînement
        ref_counts, _ = np.histogram(ref.astype(float), bins=edges)
        cur_counts, _ = np.histogram(cur.astype(float), bins=edges)
        return _psi_from_frequencies(ref_counts / len(ref), cur_counts / len(cur))

    ref_categories = ref.astype(str).unique()
    ref_freq = ref.astype(str).value_counts(normalize=True)
    cur_str = cur.astype(str)
    cur_known = cur_str[cur_str.isin(ref_categories)]
    cur_new_count = len(cur_str) - len(cur_known)
    cur_freq = cur_known.value_counts(normalize=True) if len(cur_known) else pd.Series(dtype=float)

    known_share = len(cur_known) / len(cur_str) if len(cur_str) else 1
    expected = ref_freq.reindex(ref_categories, fill_value=0.0).to_numpy()
    actual = cur_freq.reindex(ref_categories, fill_value=0.0).to_numpy() * known_share
    if cur_new_count > 0:
        expected = np.append(expected, 0.0)
        actual = np.append(actual, cur_new_count / len(cur_str))
    return _psi_from_frequencies(expected, actual)


def psi_severity(psi: float) -> str:
    """`"stable"` / `"modere"` / `"significatif"` — voir seuils Siddiqi en
    tête de module, jamais un jugement variable par variable."""
    if psi < PSI_STABLE_MAX:
        return "stable"
    if psi < PSI_MODERATE_MAX:
        return "modere"
    return "significatif"


def compute_drift_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_columns: list[str]
) -> dict:
    """Rapport de dérive complet — un PSI par variable d'entraînement
    présente dans les deux jeux de données, triées par sévérité décroissante
    (variables les plus dérivées d'abord, même convention que
    `verdict.py`/`data_quality.py` : ce qui compte le plus en tête).

    Une colonne de `feature_columns` absente de `current_df` (ex. jamais
    envoyée par aucune prédiction récente) est silencieusement omise du
    rapport plutôt que de lever une erreur — `current_df` vient de
    `Prediction.input_json` désérialisé, pas d'un schéma garanti.

    `insufficient_data=True` (moins de `MIN_CURRENT_ROWS_FOR_DRIFT`
    prédictions) : `features` reste vide plutôt que de renvoyer un PSI
    instable qui semblerait fiable sans l'être — dit honnêtement à
    l'appelant qu'il n'y a pas encore assez de recul, jamais un chiffre
    calculé quand même en silence."""
    n_current = len(current_df)
    if n_current < MIN_CURRENT_ROWS_FOR_DRIFT:
        return {
            "n_predictions_analyzed": n_current,
            "insufficient_data": True,
            "features": [],
            "n_significant": 0,
            "n_moderate": 0,
        }

    features: list[dict] = []
    for col in feature_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = compute_psi(reference_df[col], current_df[col])
        features.append({"feature": col, "psi": round(psi, 4), "severity": psi_severity(psi)})

    severity_order = {"significatif": 0, "modere": 1, "stable": 2}
    features.sort(key=lambda f: (severity_order[f["severity"]], -f["psi"]))

    n_significant = sum(1 for f in features if f["severity"] == "significatif")
    n_moderate = sum(1 for f in features if f["severity"] == "modere")
    return {
        "n_predictions_analyzed": n_current,
        "insufficient_data": False,
        "features": features,
        "n_significant": n_significant,
        "n_moderate": n_moderate,
    }
