"""Primitives statistiques partagées — association entre variables, pour
l'EDA enrichie (`services/dataset_eda.py`) et les garde-fous de données
(`services/data_quality.py`). Aucune dépendance ajoutée : tout est calculé
avec pandas/numpy/scikit-learn, déjà présents (Lot 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Association entre deux variables catégorielles — V de Cramér CORRIGÉ
    du biais de Bergsma, pas la version brute.

    La version brute du V de Cramér surestime systématiquement l'association
    quand la cardinalité est élevée ou l'échantillon petit (biais positif
    connu) : deux colonnes catégorielles indépendantes mais à beaucoup de
    catégories affichent un V brut nettement supérieur à 0, ce qui produirait
    de faux positifs de fuite cible sur ce type de colonne. La correction
    retranche un terme fonction des degrés de liberté et de la taille de
    l'échantillon avant de calculer le ratio final.

    Référence : Bergsma, W. (2013), "A bias-correction for Cramér's V and
    Tschuprow's T", Journal of the Korean Statistical Society, 42, 323-328.

    Coût : O(n) pour construire le tableau de contingence + O(r×k) pour le
    χ² (r, k = nombre de catégories de x, y) — négligeable tant que r et k
    restent bornés (voir `CATEGORICAL_CORR_MAX_UNIQUE` côté appelants, qui
    exclut les colonnes quasi-identifiants en amont).
    """
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    n = len(x)
    if n <= 1:
        return 0.0

    contingency = pd.crosstab(x, y).to_numpy(dtype=float)
    r, k = contingency.shape
    if r < 2 or k < 2:
        return 0.0  # une des deux variables ne varie pas sur l'échantillon valide

    row_sums = contingency.sum(axis=1, keepdims=True)
    col_sums = contingency.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    chi2 = float(np.sum((contingency - expected) ** 2 / expected))

    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    k_corrected = k - (k - 1) ** 2 / (n - 1)
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    denom = min(k_corrected - 1, r_corrected - 1)
    if denom <= 0:
        return 0.0  # dégénéré (échantillon trop petit face au nombre de catégories)
    return float(min(1.0, np.sqrt(phi2_corrected / denom)))


def correlation_ratio(categorical: pd.Series, numeric: pd.Series) -> float:
    """Rapport de corrélation (η, eta) — association entre une variable
    catégorielle et une variable numérique, bornée [0, 1] comme une
    corrélation classique (contrairement à l'information mutuelle brute, non
    bornée et donc pas directement comparable à un seuil de type "0.95").

    η² = variance inter-groupes / variance totale (logique d'ANOVA à un
    facteur) : proche de 1 si connaître la catégorie détermine presque
    entièrement la valeur numérique.

    Coût : O(n) — un groupby/moyenne par catégorie.
    """
    valid = categorical.notna() & numeric.notna()
    cat = categorical[valid]
    num = numeric[valid].astype(float)
    if len(num) <= 1:
        return 0.0

    overall_mean = num.mean()
    ss_total = float(((num - overall_mean) ** 2).sum())
    if ss_total == 0.0:
        return 0.0  # la numérique est constante — aucune variance à expliquer

    group_means = num.groupby(cat).mean()
    group_counts = num.groupby(cat).count()
    ss_between = float((group_counts * (group_means - overall_mean) ** 2).sum())

    eta2 = max(0.0, min(1.0, ss_between / ss_total))
    return float(np.sqrt(eta2))


def univariate_auc(feature_numeric: pd.Series, target_categorical: pd.Series) -> float:
    """Capacité d'UNE SEULE variable numérique à séparer les classes de la
    cible, utilisée directement comme score de classement (sans modèle).
    Retourne max(AUC, 1-AUC) : une variable qui sépare parfaitement mais dans
    le sens inverse (AUC≈0) est tout aussi suspecte qu'une AUC≈1.

    Binaire : AUC directe. Multiclasse : moyenne des AUC one-vs-rest (même
    logique que `roc_auc_score(..., multi_class="ovr")` déjà utilisé dans
    `ml_training.py`, mais calculée à la main ici pour appliquer
    individuellement le max(AUC, 1-AUC) par classe avant la moyenne.

    Coût : O(n log n) (tri implicite du calcul de l'AUC), une fois par classe
    en multiclasse.
    """
    valid = feature_numeric.notna() & target_categorical.notna()
    x = feature_numeric[valid].astype(float)
    y = target_categorical[valid]
    classes = y.unique()
    if len(x) < 2 or len(classes) < 2:
        return 0.5

    if len(classes) == 2:
        y_bin = (y == classes[0]).astype(int)
        try:
            auc = roc_auc_score(y_bin, x)
        except ValueError:
            return 0.5
        return float(max(auc, 1 - auc))

    y_bin = label_binarize(y, classes=classes)
    per_class_auc = []
    for i in range(y_bin.shape[1]):
        try:
            auc = roc_auc_score(y_bin[:, i], x)
        except ValueError:
            continue
        per_class_auc.append(max(auc, 1 - auc))
    return float(np.mean(per_class_auc)) if per_class_auc else 0.5


def sample_if_large(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """Échantillonnage déterministe (seed fixe, reproductible) pour borner le
    coût des calculs statistiques sur un gros dataset. Sans effet si le
    dataset tient déjà sous `max_rows`. Documenté et nommé par l'appelant
    (constante `MAX_ROWS_FOR_STATS`)."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
