"""Garde-fous de données — analyse un dataset + une colonne cible et produit
une liste d'avertissements structurés, actionnables par un utilisateur
non-data-scientist (Lot B).

Principe directeur : jamais un flag nu. Chaque avertissement porte un
niveau ("info" / "attention" / "critique"), un titre en langage clair, une
explication du POURQUOI c'est un problème, et une ACTION recommandée
concrète. Chaque détection est isolée : si l'une échoue (donnée
inattendue), elle est loguée et ignorée — jamais de propagation brute vers
l'appelant (voir `analyze_data_quality`).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd

from domains.shared.dataset_eda import compute_missing_summary
from domains.shared.ml_task import detect_task_type
from domains.shared.stats_utils import correlation_ratio, cramers_v, sample_if_large, univariate_auc

logger = logging.getLogger("datalab.data_quality")

# ── Seuils — constantes nommées, quasi-décisions produit ───────────────────
#
# Fuite cible : UN SEUIL PAR MÉTRIQUE, pas un seuil unique. Les 4 métriques
# n'ont pas la même échelle empirique pour un même niveau de "fuite réelle" :
# un V de Cramér CORRIGÉ (voir stats_utils.cramers_v) atteint rarement 0.95
# même sur une vraie fuite catégorielle imparfaite (peu de bruit mais pas une
# bijection stricte) — un seuil unique à 0.95 raterait ces cas. Valeurs de
# départ, à valider/ajuster sur des datasets réels :
TARGET_LEAKAGE_PEARSON_THRESHOLD = 0.95
# Pearson |r| — num <-> num. 0.95 signifie r²≥0.9025 (>90% de variance
# partagée) : au-delà, la relation est presque déterministe, rarement
# observé sur une vraie relation causale "normale" entre deux mesures.
TARGET_LEAKAGE_CRAMERS_V_THRESHOLD = 0.70
# V de Cramér corrigé — cat <-> cat. Nettement plus bas que 0.95 : la
# correction de biais (Bergsma) ramène déjà les fausses associations vers 0,
# donc un score qui reste ≥0.70 après correction est un signal fort, pas un
# artefact de cardinalité. Valeur de départ à confirmer sur des cas réels
# (fuite catégorielle imparfaite mais réelle).
TARGET_LEAKAGE_CORR_RATIO_THRESHOLD = 0.85
# η (rapport de corrélation) — cat <-> num. Entre les deux précédents : η²
# (part de variance numérique expliquée par la catégorie) est mécaniquement
# tiré vers le haut quand une catégorie a peu d'effectifs (comme le Cramér's
# V brut, mais sans correction de biais équivalente établie pour η) — seuil
# intermédiaire par prudence, à valider.
TARGET_LEAKAGE_AUC_THRESHOLD = 0.97
# AUC univariée (max(AUC, 1-AUC)) — num -> cat. 0.5 = variable non
# informative, 1.0 = séparation parfaite. Une seule variable numérique qui
# sépare les classes à 0.97+ à elle seule est structurellement suspecte pour
# un problème réel (rarement obtenu sans fuite ou sans que la variable SOIT
# la définition de la cible).

CLASS_IMBALANCE_RATIO_THRESHOLD = 10.0
# Ratio effectif classe majoritaire / classe minoritaire. Au-delà de 10:1,
# un modèle qui prédit toujours la classe majoritaire dépasse déjà 90%
# d'exactitude sans avoir rien appris — l'exactitude seule devient trompeuse.

CARDINALITY_RATIO_THRESHOLD = 0.9
CARDINALITY_ABSOLUTE_THRESHOLD = 50
# Cardinalité excessive (colonnes catégorielles) : au-delà de 90% de valeurs
# uniques (ratio) OU de 50 catégories distinctes (absolu, utile même sur un
# gros dataset où le ratio resterait bas), la colonne ressemble à un
# identifiant et l'encodage one-hot exploserait en autant de colonnes.

QUASI_CONSTANT_DOMINANCE_THRESHOLD = 0.99
# Catégoriel : une valeur qui représente ≥99% des lignes n'apporte
# quasiment rien pour distinguer les cas.
NUMERIC_QUASI_CONSTANT_STD_RATIO_THRESHOLD = 1e-6
# Numérique : écart-type / étendue (max-min) en-dessous de ce ratio =
# variation négligeable face à l'échelle de la colonne (plus robuste qu'un
# coefficient de variation classique std/mean, qui diverge quand la moyenne
# est proche de 0).

MIN_ROWS_THRESHOLD = 100
MIN_ROWS_PER_FEATURE_RATIO = 10
# Dataset trop petit : moins de 100 lignes, ou moins de 10 lignes par
# variable candidate — règle empirique courante en ML appliqué en-dessous
# de laquelle le risque de surapprentissage devient élevé.

HIGH_MISSING_RATE_THRESHOLD = 30.0
# % de valeurs manquantes par colonne. Aligné intentionnellement sur le
# seuil déjà utilisé (en dur) par `EdaModal.tsx` (ligne où `missing_pct >
# 30` déclenche l'icône d'alerte) — DETTE TECHNIQUE NOTÉE : ce seuil existe
# désormais en DEUX endroits (constante ici, valeur en dur côté front) à
# unifier dans un lot ultérieur (ex. exposer ce seuil via l'API plutôt que
# de le dupliquer). Non corrigé ici : hors périmètre Lot B (pas de
# modification du front existant).

COLLINEARITY_THRESHOLD = 0.9
MAX_COLLINEARITY_WARNINGS = 5
# Collinéarité entre features numériques (hors cible) : 0.9 reste
# nettement sous le seuil de fuite (0.95) mais assez haut pour ne pas noyer
# l'utilisateur — des corrélations 0.7-0.8 sont courantes et souvent
# bénignes pour des modèles d'arbres. Capé à 5 paires pour rester lisible.

MAX_ROWS_FOR_STATS = 20_000
# Bornage de coût : au-delà, les détections coûteuses (collinéarité)
# travaillent sur un échantillon déterministe (voir stats_utils.sample_if_large).
CATEGORICAL_CORR_MAX_UNIQUE = 30
# Cardinalité max pour qu'une colonne catégorielle entre dans le calcul de
# fuite cible catégorielle (Cramér's V) — au-delà, la colonne est de toute
# façon déjà couverte par le garde-fou "cardinalité excessive", et un
# tableau de contingence à forte cardinalité coûte cher pour un signal déjà
# rapporté ailleurs.

# ── Lot Nettoyage guidé des variables ───────────────────────────────────────
#
# Répond à une lacune identifiée à l'usage : les colonnes constantes/quasi-
# constantes et à cardinalité excessive étaient déjà détectées ci-dessus
# (avec une action "retirez cette colonne" en langage clair), mais rien ne
# reliait cette recommandation à une action concrète côté formulaire
# d'entraînement — l'utilisateur devait lire l'alerte puis aller décocher la
# colonne manuellement, sans lien visuel entre les deux. Les deux détections
# ci-dessous complètent le même principe ("cette colonne n'apporte rien, ou
# n'apporte rien de plus qu'une autre") ; leur exploitation concrète
# (suggestion d'exclusion actionnable) vit dans `services/feature_engineering.py`.

NUMERIC_MISTYPED_MIN_SIGNAL_RATIO = 0.5
# Part minimale de l'échantillon devant porter un signe explicite de
# formatage numérique (virgule décimale, séparateur de milliers) avant même
# de tenter un parsing — évite de confondre une colonne d'identifiants/codes
# (ex. codes postaux à zéro non significatif) avec une colonne numérique mal
# typée : un identifiant ne contient normalement ni virgule ni séparateur.
NUMERIC_MISTYPED_SUCCESS_THRESHOLD = 0.95
# Part minimale de l'échantillon qui doit effectivement se parser comme un
# nombre (une fois le signal de formatage confirmé) — même logique et même
# valeur que `DATETIME_PARSE_SUCCESS_THRESHOLD` (feature_engineering.py) :
# tolère quelques valeurs mal saisies sans les laisser dominer la décision.
NUMERIC_MISTYPED_SAMPLE_MAX_ROWS = 5_000
# Bornage de coût du test de parsing — même ordre de grandeur que
# `DATETIME_SAMPLE_MAX_ROWS`.
_THOUSANDS_CHARS = (" ", " ", "'")  # espace, espace insécable, apostrophe (convention suisse)


_LEVEL_ORDER = {"critique": 0, "attention": 1, "info": 2}


def _warning(
    level: str, code: str, title: str, explanation: str, action: str,
    columns: Optional[list[str]] = None, details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "level": level,
        "code": code,
        "title": title,
        "explanation": explanation,
        "action": action,
        "columns": columns or [],
        "details": details,
    }


def _detect_target_leakage(
    df: pd.DataFrame, target_column: str, features: list[str], task_type: str
) -> list[dict[str, Any]]:
    """Association feature -> cible, un seuil par métrique selon le type de
    chacune (voir constantes ci-dessus). Coût : O(n) par feature numérique,
    O(n) + O(r×k) par feature catégorielle (contingence bornée par
    `CATEGORICAL_CORR_MAX_UNIQUE`)."""
    warnings: list[dict[str, Any]] = []
    target = df[target_column]

    for feature in features:
        try:
            feature_series = df[feature]
            is_feature_numeric = pd.api.types.is_numeric_dtype(feature_series)

            if task_type == "regression":
                if is_feature_numeric:
                    valid = feature_series.notna() & target.notna()
                    if valid.sum() < 2 or feature_series[valid].std() == 0 or target[valid].std() == 0:
                        continue
                    value = abs(float(np.corrcoef(feature_series[valid], target[valid])[0, 1]))
                    metric, threshold = "pearson", TARGET_LEAKAGE_PEARSON_THRESHOLD
                else:
                    if feature_series.nunique() > CATEGORICAL_CORR_MAX_UNIQUE:
                        continue
                    value = correlation_ratio(feature_series, target)
                    metric, threshold = "correlation_ratio", TARGET_LEAKAGE_CORR_RATIO_THRESHOLD
            else:  # classification
                if is_feature_numeric:
                    value = univariate_auc(feature_series, target)
                    metric, threshold = "auc", TARGET_LEAKAGE_AUC_THRESHOLD
                else:
                    if feature_series.nunique() > CATEGORICAL_CORR_MAX_UNIQUE:
                        continue
                    value = cramers_v(feature_series, target)
                    metric, threshold = "cramers_v", TARGET_LEAKAGE_CRAMERS_V_THRESHOLD

            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            if value > threshold:
                warnings.append(_warning(
                    level="critique",
                    code="fuite_cible",
                    title=f"« {feature} » prédit presque parfaitement la cible",
                    explanation=(
                        "Cette colonne est si fortement liée à la cible qu'elle contient "
                        "peut-être l'information que vous cherchez à prédire — ou une "
                        "variable calculée APRÈS l'événement à prédire (une conséquence, "
                        "pas une cause)."
                    ),
                    action=(
                        "Vérifiez que cette colonne sera bien disponible AVANT de faire "
                        "une prédiction en conditions réelles. Si elle ne l'est pas, ou "
                        "si elle découle directement de la cible, retirez-la des "
                        "variables utilisées pour l'entraînement."
                    ),
                    columns=[feature],
                    details={"metric": metric, "value": round(float(value), 4), "threshold": threshold},
                ))
        except Exception:
            logger.warning("[DataQuality] Détection de fuite échouée pour la colonne '%s'", feature, exc_info=True)

    return warnings


def _detect_class_imbalance(df: pd.DataFrame, target_column: str, task_type: str) -> list[dict[str, Any]]:
    """Coût : O(n), un value_counts sur la cible."""
    if task_type != "classification":
        return []
    counts = df[target_column].value_counts(dropna=True)
    if len(counts) < 2 or counts.min() == 0:
        return []
    ratio = float(counts.max() / counts.min())
    if ratio <= CLASS_IMBALANCE_RATIO_THRESHOLD:
        return []
    return [_warning(
        level="attention",
        code="desequilibre_classes",
        title="Classes très déséquilibrées",
        explanation=(
            f"La classe la plus fréquente apparaît environ {ratio:.0f} fois plus "
            "souvent que la plus rare. Un modèle qui prédirait toujours la classe "
            "majoritaire afficherait déjà une exactitude élevée sans avoir rien appris."
        ),
        action=(
            "Regardez le F1-score ou le rappel par classe plutôt que la seule "
            "exactitude globale pour juger le modèle. Un rééquilibrage "
            "(sur-échantillonnage, pondération des classes) pourra être proposé "
            "lors de l'entraînement."
        ),
        columns=[target_column],
        details={"ratio": round(ratio, 2)},
    )]


def _detect_high_cardinality(df: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    """Colonnes catégorielles uniquement (l'encodage one-hot ne concerne pas
    le numérique). Coût : O(n) par colonne (nunique).

    `n_estimated_onehot_columns` (transparence) : nombre de colonnes que
    produirait un encodage one-hot de cette colonne — informatif seulement.
    Le moteur (`services/ml_training.py`, diagnostic "bad allocation")
    préserve désormais le résultat sous forme SPARSE jusqu'au fit/predict
    pour le catalogue par défaut (100% modèles à arbres), donc un nombre
    élevé ici n'implique plus une explosion mémoire dense comme avant ce
    correctif — la recommandation reste de retirer un identifiant, mais pour
    son absence de valeur prédictive, pas (plus) par crainte de saturation."""
    warnings: list[dict[str, Any]] = []
    for col in features:
        try:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                continue
            n = int(series.notna().sum())
            if n == 0:
                continue
            n_unique = int(series.nunique())
            ratio = n_unique / n
            if ratio <= CARDINALITY_RATIO_THRESHOLD and n_unique <= CARDINALITY_ABSOLUTE_THRESHOLD:
                continue
            warnings.append(_warning(
                level="attention",
                code="cardinalite_excessive",
                title=f"« {col} » a un très grand nombre de catégories différentes",
                explanation=(
                    f"{n_unique} valeurs différentes pour {n} lignes — cela ressemble "
                    "davantage à un identifiant (numéro de client, référence...) "
                    "qu'à une variable réellement prédictive."
                ),
                action=(
                    "Si c'est bien un identifiant, retirez-le des variables utilisées : "
                    "il n'apporte aucune valeur prédictive, quelle que soit sa taille."
                ),
                columns=[col],
                details={
                    "n_unique": n_unique, "n_rows": n, "ratio": round(ratio, 4),
                    "n_estimated_onehot_columns": n_unique,
                },
            ))
        except Exception:
            logger.warning("[DataQuality] Détection de cardinalité échouée pour la colonne '%s'", col, exc_info=True)
    return warnings


def _detect_constant_columns(df: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    """Coût : O(n) par colonne (nunique/std, ou value_counts pour le
    catégoriel)."""
    warnings: list[dict[str, Any]] = []
    for col in features:
        try:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            is_constant = series.nunique() <= 1
            is_quasi = False
            if not is_constant:
                if pd.api.types.is_numeric_dtype(series):
                    value_range = float(series.max() - series.min())
                    if value_range > 0:
                        is_quasi = (float(series.std()) / value_range) < NUMERIC_QUASI_CONSTANT_STD_RATIO_THRESHOLD
                else:
                    top_share = float(series.value_counts(normalize=True).iloc[0])
                    is_quasi = top_share >= QUASI_CONSTANT_DOMINANCE_THRESHOLD

            if not (is_constant or is_quasi):
                continue
            label = "constante" if is_constant else "quasi constante"
            warnings.append(_warning(
                level="info",
                code="colonne_constante",
                title=f"« {col} » est {label}",
                explanation=(
                    "Cette colonne a (quasiment) toujours la même valeur — elle "
                    "n'apporte aucune information utile pour distinguer les cas."
                ),
                action="Vous pouvez la retirer des variables utilisées, sans perte d'information.",
                columns=[col],
            ))
        except Exception:
            logger.warning("[DataQuality] Détection de colonne constante échouée pour '%s'", col, exc_info=True)
    return warnings


def _detect_small_dataset(df: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    """Coût : O(1) — comptages déjà disponibles."""
    warnings: list[dict[str, Any]] = []
    n_rows = len(df)
    n_features = len(features)

    if n_rows < MIN_ROWS_THRESHOLD:
        warnings.append(_warning(
            level="attention",
            code="dataset_trop_petit",
            title="Peu de lignes dans ce dataset",
            explanation=(
                f"Seulement {n_rows} lignes au total — les résultats (métriques, "
                "importance des variables) risquent d'être peu fiables et de mal "
                "généraliser à de nouvelles données."
            ),
            action="Interprétez les résultats avec prudence, ou complétez le dataset si possible.",
            details={"n_rows": n_rows},
        ))

    if n_features > 0 and (n_rows / n_features) < MIN_ROWS_PER_FEATURE_RATIO:
        warnings.append(_warning(
            level="attention",
            code="ratio_lignes_variables_faible",
            title="Peu de lignes par rapport au nombre de variables",
            explanation=(
                f"{n_rows} lignes pour {n_features} variables — le modèle a "
                "statistiquement plus de variables que d'exemples pour apprendre "
                "à les utiliser, ce qui favorise le surapprentissage."
            ),
            action="Réduisez le nombre de variables utilisées, ou complétez le dataset si possible.",
            details={"n_rows": n_rows, "n_features": n_features},
        ))

    return warnings


def _detect_high_missing_rate(df: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    """Réutilise `compute_missing_summary` (déjà O(n) sur tout le dataset,
    calculé une seule fois) plutôt que de recalculer colonne par colonne."""
    warnings: list[dict[str, Any]] = []
    features_set = set(features)
    for entry in compute_missing_summary(df):
        if entry["column"] not in features_set:
            continue
        if entry["missing_pct"] is None or entry["missing_pct"] <= HIGH_MISSING_RATE_THRESHOLD:
            continue
        warnings.append(_warning(
            level="attention",
            code="valeurs_manquantes_elevees",
            title=f"« {entry['column']} » a beaucoup de valeurs manquantes",
            explanation=(
                f"Environ {entry['missing_pct']:.0f}% des valeurs manquent sur cette "
                "colonne. Au-delà d'un certain taux, le remplacement automatique "
                "(médiane, valeur la plus fréquente) devient peu fiable et peut "
                "introduire un biais."
            ),
            action=(
                "Vérifiez pourquoi ces valeurs manquent. Envisagez de retirer la "
                "colonne si le taux est très élevé, ou conservez-la en gardant "
                "cette limite à l'esprit."
            ),
            columns=[entry["column"]],
            details={"missing_pct": round(float(entry["missing_pct"]), 2)},
        ))
    return warnings


def _detect_collinearity(df: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    """Coût : O(n × k²) pour la matrice de corrélation (k = nb de features
    numériques) — échantillonné via `sample_if_large` au-delà de
    `MAX_ROWS_FOR_STATS` lignes."""
    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_features) < 2:
        return []

    sampled = sample_if_large(df[numeric_features], MAX_ROWS_FOR_STATS)
    corr = sampled.corr()

    pairs: list[tuple[str, str, float]] = []
    for i in range(len(numeric_features)):
        for j in range(i + 1, len(numeric_features)):
            c1, c2 = numeric_features[i], numeric_features[j]
            r = corr.loc[c1, c2]
            if pd.notna(r) and abs(float(r)) > COLLINEARITY_THRESHOLD:
                pairs.append((c1, c2, float(r)))
    pairs.sort(key=lambda p: -abs(p[2]))

    return [
        _warning(
            level="info",
            code="collinearite_forte",
            title=f"« {c1} » et « {c2} » sont très corrélées",
            explanation=(
                "Ces deux variables portent une information très redondante — "
                "connaître l'une donne presque la valeur de l'autre."
            ),
            action="Vous pouvez généralement en retirer une des deux sans perte de performance notable.",
            columns=[c1, c2],
            details={"correlation": round(r, 4)},
        )
        for c1, c2, r in pairs[:MAX_COLLINEARITY_WARNINGS]
    ]


def _detect_duplicate_columns(df: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    """Colonnes de contenu strictement identique sous un nom différent —
    redondance à 100%, distincte de la colinéarité (`_detect_collinearity`,
    Pearson ≥ 0.9, numérique uniquement, niveau "info") : ici la comparaison
    est une égalité exacte valeur par valeur, numérique OU catégorielle, et le
    niveau est "attention" — un doublon exact ne laisse aucune ambiguïté sur
    l'action à prendre, contrairement à une simple corrélation forte.

    Coût : O(n) par colonne pour un hash (`pandas.util.hash_pandas_object`),
    puis comparaison exacte (`Series.equals`, seule fiable à 100% — un hash
    identique ne garantit pas une égalité, l'inverse si) seulement entre
    colonnes de même hash. Bien moins cher qu'une comparaison `equals` O(k²)
    systématique sur un dataset à beaucoup de colonnes : un hash différent
    élimine déjà la quasi-totalité des paires possibles sans les comparer."""
    warnings: list[dict[str, Any]] = []
    hash_groups: dict[int, list[str]] = defaultdict(list)
    for col in features:
        try:
            h = int(pd.util.hash_pandas_object(df[col], index=False).sum())
            hash_groups[h].append(col)
        except Exception:
            logger.warning("[DataQuality] Hash de colonne échoué pour '%s'", col, exc_info=True)

    reported: set[str] = set()
    for cols in hash_groups.values():
        if len(cols) < 2:
            continue
        for i in range(len(cols)):
            c1 = cols[i]
            if c1 in reported:
                continue
            for j in range(i + 1, len(cols)):
                c2 = cols[j]
                if c2 in reported:
                    continue
                try:
                    if not df[c1].equals(df[c2]):
                        continue
                    warnings.append(_warning(
                        level="attention",
                        code="colonnes_dupliquees",
                        title=f"« {c1} » et « {c2} » ont un contenu strictement identique",
                        explanation=(
                            "Ces deux colonnes contiennent exactement les mêmes valeurs, "
                            "ligne par ligne — l'une est une copie ou un renommage de "
                            "l'autre, elle n'apporte aucune information supplémentaire."
                        ),
                        action=f"Retirez « {c2} » (ou « {c1} ») des variables utilisées : garder les deux n'apporte rien.",
                        columns=[c1, c2],
                    ))
                    reported.add(c2)
                except Exception:
                    logger.warning(
                        "[DataQuality] Détection de doublon échouée pour '%s'/'%s'", c1, c2, exc_info=True
                    )
    return warnings


def _try_parse_numeric_text(text: str) -> Optional[float]:
    """Tente de parser une chaîne comme un nombre écrit avec une convention
    différente du point décimal standard : séparateur de milliers (espace,
    espace insécable, apostrophe) et/ou virgule décimale (convention FR/EU).
    Retourne `None` si la chaîne ne ressemble à aucune de ces conventions —
    jamais d'exception, cœur du parsing réutilisé par `services/
    feature_engineering.py` pour appliquer la conversion réellement approuvée.
    """
    cleaned = text.strip()
    if not cleaned:
        return None
    for ch in _THOUSANDS_CHARS:
        cleaned = cleaned.replace(ch, "")
    # Une seule virgule -> décimale (convention FR/EU) ; un point restant
    # avant la virgule est alors un séparateur de milliers (ex. "1.234,56").
    if cleaned.count(",") == 1:
        integer_part, _, decimal_part = cleaned.partition(",")
        cleaned = f"{integer_part.replace('.', '')}.{decimal_part}"
    try:
        return float(cleaned)
    except ValueError:
        return None


def _has_numeric_format_signal(text: str) -> bool:
    """Vrai si la chaîne porte un signe explicite de formatage numérique
    (virgule ou séparateur de milliers) — sert de garde-fou avant même de
    tenter un parsing (voir `NUMERIC_MISTYPED_MIN_SIGNAL_RATIO`) : une
    colonne d'identifiants/codes (ex. codes postaux à zéro non significatif)
    ne porte normalement aucun de ces signes et ne doit jamais être proposée
    à la conversion numérique."""
    return "," in text or any(ch in text for ch in _THOUSANDS_CHARS)


def _detect_mistyped_numeric(df: pd.DataFrame, features: list[str]) -> list[dict[str, Any]]:
    """Colonnes texte qui sont en réalité des nombres mal formatés (virgule
    décimale et/ou séparateur de milliers) — un cas courant sur des exports
    métier réels, invisible aujourd'hui : la colonne est alors traitée comme
    catégorielle (chaque valeur = une modalité distincte) au lieu d'une
    variable numérique exploitable. Deux garde-fous avant de signaler,
    volontairement stricts pour ne jamais confondre avec une colonne
    d'identifiants (qui, elle, doit rester du texte) :
    1. une part suffisante de l'échantillon doit porter un signe explicite de
       formatage numérique (`_has_numeric_format_signal`) — un simple entier
       sans séparateur ("00123") n'est jamais un candidat ;
    2. parmi ces candidats, une part suffisante doit effectivement se parser
       comme un nombre (`_try_parse_numeric_text`).

    Coût : O(n) par colonne texte, borné par échantillonnage déterministe
    (`NUMERIC_MISTYPED_SAMPLE_MAX_ROWS`, même patron que la détection
    datetime de `services/feature_engineering.py`)."""
    warnings: list[dict[str, Any]] = []
    for col in features:
        try:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                continue
            non_null = series.dropna().astype(str)
            if len(non_null) == 0:
                continue
            sampled = sample_if_large(non_null.to_frame("v"), NUMERIC_MISTYPED_SAMPLE_MAX_ROWS)["v"]

            signal_ratio = float(sampled.map(_has_numeric_format_signal).mean())
            if signal_ratio < NUMERIC_MISTYPED_MIN_SIGNAL_RATIO:
                continue

            parsed = sampled.map(_try_parse_numeric_text)
            success_ratio = float(parsed.notna().mean())
            if success_ratio < NUMERIC_MISTYPED_SUCCESS_THRESHOLD:
                continue

            warnings.append(_warning(
                level="attention",
                code="numerique_mal_type",
                title=f"« {col} » ressemble à une variable numérique stockée en texte",
                explanation=(
                    "Cette colonne contient presque uniquement des nombres, mais "
                    "écrits avec une virgule décimale ou un séparateur de milliers — "
                    "elle est donc traitée comme une catégorie (chaque valeur devient "
                    "une modalité distincte) au lieu d'une vraie variable numérique."
                ),
                action=f"Convertissez « {col} » en nombre (une suggestion automatique est proposée à l'étape suivante).",
                columns=[col],
                details={"parse_success_ratio": round(success_ratio, 4)},
            ))
        except Exception:
            logger.warning("[DataQuality] Détection numérique mal typé échouée pour '%s'", col, exc_info=True)
    return warnings


def _group_column_transparency_warning(group_column: str) -> dict[str, Any]:
    return _warning(
        level="info",
        code="colonne_groupe_exclue",
        title=f"« {group_column} » est utilisée comme groupe anti-fuite",
        explanation=(
            "Cette colonne identifie des échantillons répétés (ex. plusieurs "
            "mesures liées) et sert uniquement à séparer proprement "
            "l'entraînement et le test — elle n'est pas analysée ci-dessus comme "
            "variable prédictive."
        ),
        action="Aucune action nécessaire, c'est le comportement attendu.",
        columns=[group_column],
    )


def analyze_data_quality(
    df: pd.DataFrame, target_column: Optional[str] = None, group_column: Optional[str] = None
) -> list[dict[str, Any]]:
    """Point d'entrée — analyse `df` par rapport à `target_column` et
    retourne une liste d'avertissements triés (critique > attention > info).

    `target_column` optionnel (Lot Nettoyage guidé des variables) : absent,
    les détections qui exigent une cible (fuite, déséquilibre de classes)
    sont simplement omises — toutes les détections STRUCTURELLES (colonnes
    constantes, cardinalité excessive, doublons exacts, numérique mal typé,
    valeurs manquantes, colinéarité, dataset trop petit) restent actives.
    Permet une analyse de qualité dès l'exploration d'un dataset (`GET
    /datasets/{id}/eda`), avant même de choisir une cible pour un
    entraînement — l'utilisateur n'a plus besoin d'atteindre l'étape
    "qualité des données" du formulaire pour voir qu'une colonne est
    inutile. Rétrocompatible : tout appelant existant qui fournit déjà
    `target_column` conserve exactement le même comportement.

    `group_column`, si fourni, est EXCLU de l'ensemble des colonnes
    analysées comme features (calculé UNE SEULE FOIS ici — `features_to_
    analyze` — et consommé tel quel par toutes les détections, pour garantir
    qu'aucune ne l'analyse par erreur) : ce n'est pas une variable
    prédictive, aucune alerte de qualité de feature n'a de sens dessus. Un
    avertissement "info" signale explicitement cette exclusion.

    Lève `KeyError` si `target_column` est fourni mais n'existe pas dans `df`
    (à l'appelant de traduire en erreur HTTP, comme pour
    `compute_histogram`). Toute autre exception issue d'une détection
    individuelle est capturée et loguée — jamais propagée : le pire résultat
    possible est une liste partielle ou vide, jamais un crash.
    """
    if target_column is not None and target_column not in df.columns:
        raise KeyError(f"Colonne cible '{target_column}' absente du dataset")

    has_group = group_column is not None and group_column in df.columns
    excluded = ({target_column} if target_column is not None else set()) | ({group_column} if has_group else set())
    features_to_analyze = [c for c in df.columns if c not in excluded]

    task_type: Optional[str] = None
    if target_column is not None:
        try:
            task_type = detect_task_type(df[target_column])
        except Exception:
            logger.warning("[DataQuality] Détection du type de tâche échouée sur '%s'", target_column, exc_info=True)
            task_type = None  # dégradé : détections structurelles conservées, fuite/déséquilibre omises ci-dessous

    warnings: list[dict[str, Any]] = []
    detections = [
        lambda: _detect_high_cardinality(df, features_to_analyze),
        lambda: _detect_constant_columns(df, features_to_analyze),
        lambda: _detect_duplicate_columns(df, features_to_analyze),
        lambda: _detect_mistyped_numeric(df, features_to_analyze),
        lambda: _detect_small_dataset(df, features_to_analyze),
        lambda: _detect_high_missing_rate(df, features_to_analyze),
        lambda: _detect_collinearity(df, features_to_analyze),
    ]
    if target_column is not None and task_type is not None:
        detections = [
            lambda: _detect_target_leakage(df, target_column, features_to_analyze, task_type),
            lambda: _detect_class_imbalance(df, target_column, task_type),
        ] + detections
    for detect in detections:
        try:
            warnings.extend(detect())
        except Exception:
            logger.warning("[DataQuality] Une détection a échoué et a été ignorée", exc_info=True)

    if has_group:
        warnings.append(_group_column_transparency_warning(group_column))  # type: ignore[arg-type]

    warnings.sort(key=lambda w: _LEVEL_ORDER.get(w["level"], 99))
    return warnings
