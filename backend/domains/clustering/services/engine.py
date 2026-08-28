"""Moteur d'entraînement du clustering — Lot 11 (ML non supervisé).

Module séparé de `ml_training.py` par construction (voir
`clustering_registry.py` pour le raisonnement complet) — aucune donnée ni
fonction partagée au-delà de `services/ml_preprocessing.py::build_preprocessor`,
générique et déjà indépendant de toute notion de cible.

Suit le même principe que le Lot D (leaderboard supervisé) : plusieurs
configurations sont comparées automatiquement (plusieurs k, plusieurs
algorithmes), jamais un seul essai lancé à l'aveugle — voir
`services/data-science.md` du skill senior-ai-saas-engineer, section
"Sélection du nombre de clusters : assistée... comparer plusieurs
configurations avant de choisir"."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score

from domains.clustering.services.registry import CLUSTER_REGISTRY, resolve_dbscan_eps, specs_for
from domains.shared.ml_preprocessing import TrainingAbortedError, build_preprocessor
from domains.shared.stats_utils import sample_if_large

ProgressCallback = Callable[[str, int], None]

# Silhouette n'est calculée que sur les points NON-bruit (voir
# `_compute_cluster_metrics`) — un DBSCAN qui ne rattache qu'une poignée de
# points très compacts à un cluster et classe tout le reste en bruit peut
# afficher une silhouette proche de 1 tout en ne structurant presque rien du
# dataset. Un candidat au-delà de ce budget de bruit n'est jamais élu
# gagnant sur ce seul critère (AUDIT_PILIER2_ET_REFONTE_UX.md, P2) — il reste
# visible dans le classement (transparence), juste relégué après les
# candidats qui respectent le budget.
MAX_SELECTABLE_NOISE_RATIO = 0.5

# Même borne que `dimensionality_training.py::MAX_ROWS_FOR_EMBEDDING` (pas
# `MAX_ROWS_FOR_ANOMALY`, plus généreuse : ce registre inclut le clustering
# hiérarchique, dont le linkage de Ward est O(n²) en mémoire — Isolation
# Forest/LOF, eux, restent efficaces à bien plus grande échelle). Sans ce
# plafond, un dataset volumineux pouvait déclencher un MemoryError en cours
# de job (Lot 6B, §F.2 — transparence d'échantillonnage manquante).
#
# Retour utilisateur direct : "l'entreprise vient avec 50 000 lignes, la
# plateforme n'en traite que 5 000 (10 %), est-ce que ça généralise ?" — sur
# le plan statistique pur la réponse est oui (un tirage aléatoire simple de
# 5 000 lignes donne une précision quasi identique à 50 %/80 % : l'erreur
# d'estimation dépend de la taille ABSOLUE de l'échantillon, pas de la
# fraction couverte, tant que n reste petit devant N — Cochran, théorie de
# l'échantillonnage). Le vrai facteur limitant n'est donc pas la précision
# mais le coût mémoire des algorithmes O(n²) du registre (hiérarchique
# TOUJOURS, DBSCAN potentiellement selon la dimensionnalité post-encodage).
# D'où un plafond DIFFÉRENCIÉ : généreux si la comparaison ne porte que sur
# les algorithmes de partitionnement (KMeans/MiniBatchKMeans, coût quasi
# linéaire — voir `_effective_row_cap`), conservateur dès que hiérarchique
# ou DBSCAN sont sélectionnés.
MAX_ROWS_FOR_CLUSTERING = 5000
MAX_ROWS_FOR_CLUSTERING_LINEAR_ONLY = 20_000
_LINEAR_ONLY_ALGORITHM_IDS = {"kmeans", "minibatch_kmeans"}
_ROW_INDEX_COLUMN = "__cl_row_index__"


def _effective_row_cap(algorithm_ids: list[str] | None) -> int:
    """Plafond de lignes appliqué à ce job — voir le commentaire de
    `MAX_ROWS_FOR_CLUSTERING` ci-dessus. `None`/vide = registre par défaut,
    qui inclut hiérarchique -> plafond conservateur."""
    if algorithm_ids and set(algorithm_ids) <= _LINEAR_ONLY_ALGORITHM_IDS:
        return MAX_ROWS_FOR_CLUSTERING_LINEAR_ONLY
    return MAX_ROWS_FOR_CLUSTERING

# Indicateur de stabilité de k (Lot 6B, §F.2) — par SOUS-ÉCHANTILLONNAGE
# (family-agnostic, contrairement à une comparaison multi-seed qui n'aurait
# aucun sens pour DBSCAN/hiérarchique, déterministes par construction) :
# ajuste le même algorithme sur plusieurs sous-ensembles aléatoires et mesure
# l'accord (ARI) entre les étiquettes obtenues sur les points communs à deux
# sous-ensembles consécutifs. Borné indépendamment de MAX_ROWS_FOR_CLUSTERING
# — l'estimation de stabilité n'a pas besoin de la totalité de l'échantillon
# principal, et hiérarchique (O(n²)) rendrait N_STABILITY_ROUNDS refits
# coûteux sur un sous-échantillon déjà grand.
N_STABILITY_ROUNDS = 5
STABILITY_SUBSAMPLE_FRACTION = 0.8
MAX_ROWS_FOR_STABILITY = 1000
MIN_ROWS_FOR_STABILITY = 20


@dataclass
class ClusteringConfig:
    # `None` = mode guidé (sous-ensemble par défaut du registre) ; liste
    # explicite = mode expert, même pattern que `TrainingConfig.model_ids`
    # côté supervisé (Lot E2).
    algorithm_ids: Optional[list[str]] = None
    seed: int = 42


@dataclass
class ClusterCandidateResult:
    algorithm_id: str
    label: str
    family: str
    params: dict[str, Any]
    n_clusters: int
    silhouette: Optional[float]
    davies_bouldin: Optional[float]
    calinski_harabasz: Optional[float]
    noise_ratio: float
    is_winner: bool
    rank: int
    # Rangs individuels + rang composite (voir `_attach_composite_rank`) —
    # `None` pour les candidats hors budget de bruit (jamais classés, voir
    # `_rank_candidates_with_noise_budget`). Exposés pour que la sélection
    # ne soit jamais une boîte noire côté UI : l'utilisateur voit POURQUOI
    # le gagnant a été choisi (ex. "1er en silhouette, 2e en Davies-Bouldin,
    # 6e en Calinski-Harabasz — rang composite 3.0"), jamais juste une
    # affirmation.
    # Floats, pas des entiers — une égalité entre 2 candidats sur une
    # métrique produit un rang moyen fractionnaire (voir `_rank_with_ties`,
    # ex. 2.5 pour 2 candidats à égalité aux rangs 2 et 3).
    rank_silhouette: Optional[float] = None
    rank_davies_bouldin: Optional[float] = None
    rank_calinski_harabasz: Optional[float] = None
    composite_rank: Optional[float] = None
    # Profil complet (segments, variables différenciantes) — calculé
    # UNIQUEMENT pour le top 3 (retour utilisateur : "propose les 3
    # meilleurs modèles, résultats propres pour chaque, laisse le choix à
    # l'utilisateur"), jamais pour les ~10 autres candidats du classement
    # (coût de calcul + volume de réponse non justifiés pour des
    # configurations que l'utilisateur ne consultera probablement jamais en
    # détail). `None` au-delà du top 3.
    cluster_profiles: Optional[list["ClusterProfile"]] = None
    noise_count: Optional[int] = None


@dataclass
class ClusterProfile:
    cluster_id: int
    size: int
    size_pct: float
    numeric_summary: dict[str, dict[str, float]]
    categorical_summary: dict[str, dict[str, Any]]
    # Variables numériques triées par |z-score| décroissant — ce qui
    # distingue le plus ce cluster de la population globale, pas seulement
    # ses propres statistiques (voir skill senior-ai-saas-engineer,
    # data-science.md : "variables différenciantes... pas seulement les
    # stats propres au cluster").
    differentiating_variables: list[str]


@dataclass
class ClusteringResult:
    winning_algorithm_id: str
    winning_label: str
    all_candidates: list[ClusterCandidateResult]
    labels: list[int]  # cluster assigné par ligne du dataset d'origine (-1 = bruit, DBSCAN)
    noise_count: int
    cluster_profiles: list[ClusterProfile]
    feature_columns: list[str]
    model_card: dict[str, Any]
    pipeline_bundle: dict[str, Any] = field(repr=False)


def _compute_cluster_metrics(X_processed: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Silhouette / Davies-Bouldin / Calinski-Harabasz calculés en EXCLUANT
    le bruit DBSCAN (label -1) — un point de bruit n'appartient à aucun
    cluster réel, l'inclure fausserait les trois métriques. `None` si moins
    de 2 clusters distincts restent (dégénéré, jamais une exception —
    filtré ensuite par l'appelant, même logique de dégradation propre que
    `services/ml_training.py`)."""
    labels = np.asarray(labels)
    noise_mask = labels == -1
    noise_ratio = float(noise_mask.mean())
    core_labels = labels[~noise_mask]
    core_X = X_processed[~noise_mask]
    n_clusters = len(np.unique(core_labels))

    if n_clusters < 2 or n_clusters >= len(core_labels):
        return {
            "n_clusters": n_clusters,
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "noise_ratio": noise_ratio,
        }
    return {
        "n_clusters": n_clusters,
        "silhouette": float(silhouette_score(core_X, core_labels)),
        "davies_bouldin": float(davies_bouldin_score(core_X, core_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(core_X, core_labels)),
        "noise_ratio": noise_ratio,
    }


def _build_cluster_profiles(X: pd.DataFrame, labels: np.ndarray) -> list[ClusterProfile]:
    """Profil interprétable par cluster, calculé sur les données D'ORIGINE
    (pas préprocessées — une moyenne sur une colonne one-hot ne veut rien
    dire pour un utilisateur non-DS). Jamais de profil pour le bruit DBSCAN
    (-1) : ce n'est pas un segment, juste des points non rattachés."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    global_mean = X[numeric_cols].mean() if numeric_cols else pd.Series(dtype=float)
    global_std = X[numeric_cols].std() if numeric_cols else pd.Series(dtype=float)
    # Fréquence de chaque valeur catégorielle sur l'ENSEMBLE de la population
    # (pas seulement au sein du cluster) — même esprit que le z-score
    # numérique ci-dessus : distingue une valeur dominante réellement
    # caractéristique du cluster d'une valeur simplement fréquente partout
    # (Lot 6B, §F.2 — transparence catégorielle, absente jusqu'ici pour le
    # clustering : `top_pct` seul, sans référence à la population globale).
    global_category_pct: dict[str, dict[str, float]] = {
        col: {str(idx): float(pct) for idx, pct in X[col].value_counts(normalize=True).items()}
        for col in categorical_cols
    }

    n_total = len(X)
    unique_labels = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    profiles: list[ClusterProfile] = []

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        subset = X[mask]
        size = int(mask.sum())

        numeric_summary: dict[str, dict[str, float]] = {}
        z_scores: dict[str, float] = {}
        for col in numeric_cols:
            mean = float(subset[col].mean()) if size > 0 else 0.0
            median = float(subset[col].median()) if size > 0 else 0.0
            std = global_std.get(col, 0.0)
            z = (mean - global_mean.get(col, 0.0)) / std if std and not np.isnan(std) and std != 0 else 0.0
            numeric_summary[col] = {"mean": mean, "median": median, "z_score": float(z)}
            z_scores[col] = abs(float(z))

        categorical_summary: dict[str, dict[str, Any]] = {}
        for col in categorical_cols:
            counts = subset[col].value_counts()
            if len(counts) > 0 and size > 0:
                top_category = str(counts.index[0])
                top_pct = float(counts.iloc[0] / size * 100)
                population_pct = global_category_pct.get(col, {}).get(top_category, 0.0) * 100
                categorical_summary[col] = {
                    "top_category": top_category,
                    "top_pct": top_pct,
                    "population_pct": population_pct,
                    # Sur-représentation dans ce cluster vs le reste du
                    # dataset — `None` (jamais 0 ni une division par zéro)
                    # quand la catégorie est absente ailleurs, dégradation
                    # honnête plutôt qu'un ratio infini inventé.
                    "lift": (top_pct / population_pct) if population_pct > 0 else None,
                }

        differentiating = sorted(z_scores, key=lambda c: z_scores[c], reverse=True)[:5]

        profiles.append(
            ClusterProfile(
                cluster_id=cluster_id,
                size=size,
                size_pct=float(size / n_total * 100) if n_total else 0.0,
                numeric_summary=numeric_summary,
                categorical_summary=categorical_summary,
                differentiating_variables=differentiating,
            )
        )
    return profiles


def _rank_with_ties(values: list[float], descending: bool) -> list[float]:
    """Rang MOYEN (1 = meilleur), à égalité pour des valeurs strictement
    égales — jamais un rang arbitrairement départagé par l'ordre
    d'apparition dans la liste d'entrée.

    Bug réel trouvé en testant ce module (`_attach_composite_rank`, avant
    ce correctif) : une simple énumération après `sorted()` (stable)
    attribuait des rangs CONSÉCUTIFS (1, 2...) à des valeurs strictement
    égales, dans l'ordre où elles apparaissaient dans la liste d'entrée —
    un candidat cité en premier gagnait ainsi un avantage de rang sur les
    métriques à égalité, sans AUCUNE différence réelle de qualité. Méthode
    du rang moyen (« average » de `scipy.stats.rankdata`, jamais réimportée
    ici pour rester sans dépendance nouvelle sur une fonction de quelques
    lignes) : des valeurs à égalité se partagent la moyenne des rangs
    qu'elles auraient occupés (ex. 2 candidats à égalité pour les rangs 2
    et 3 reçoivent chacun 2.5), neutre par construction vis-à-vis de
    l'ordre d'entrée."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i], reverse=descending)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2  # rangs 1-indexés i+1..j+1, moyennés
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _attach_composite_rank(candidates: list[dict[str, Any]]) -> None:
    """Rang composite (moyenne de Borda, avec gestion des égalités — voir
    `_rank_with_ties`) sur les 3 métriques de qualité — silhouette (plus
    haut = meilleur), Davies-Bouldin (plus bas = meilleur), Calinski-
    Harabasz (plus haut = meilleur). Chaque candidat reçoit un rang sur
    CHAQUE métrique, moyenné en `composite_rank` (plus bas = meilleur dans
    les trois à la fois).

    Retour utilisateur direct, observé en usage réel (DBSCAN puis K-Means) :
    la sélection au seul silhouette pouvait élire une configuration classée
    dans le dernier tiers en Davies-Bouldin (clusters larges et peu denses)
    pour un gain de silhouette marginal sur une configuration nettement
    meilleure sur les deux autres métriques. Pratique standard de
    sélection multi-critères non supervisée (agrégation de rangs / méthode
    de Borda), pas une heuristique inventée pour ce projet — MAIS ne
    change pas systématiquement le résultat : si les 3 métriques ne
    s'accordent pas sur un même « meilleur compromis » clair (ex. l'une
    favorise nettement le gagnant au silhouette), le rang composite peut
    retomber en égalité, départagée par la silhouette (voir
    `_rank_candidates_with_noise_budget`) — jamais un remplacement aveugle
    d'une métrique unique par une autre, un vrai arbitrage entre les
    trois à la fois."""
    n = len(candidates)
    rank_silhouette = _rank_with_ties([c["silhouette"] for c in candidates], descending=True)
    rank_db = _rank_with_ties([c["davies_bouldin"] for c in candidates], descending=False)
    rank_ch = _rank_with_ties([c["calinski_harabasz"] for c in candidates], descending=True)
    for i in range(n):
        candidates[i]["rank_silhouette"] = rank_silhouette[i]
        candidates[i]["rank_davies_bouldin"] = rank_db[i]
        candidates[i]["rank_calinski_harabasz"] = rank_ch[i]
        candidates[i]["composite_rank"] = (rank_silhouette[i] + rank_db[i] + rank_ch[i]) / 3


def _rank_candidates_with_noise_budget(
    valid: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Classe les candidats sur le RANG COMPOSITE (voir `_attach_composite_rank`
    — moyenne des rangs silhouette/Davies-Bouldin/Calinski-Harabasz, jamais
    la silhouette seule depuis ce correctif), en excluant du sommet du
    classement ceux dont le `noise_ratio` dépasse `MAX_SELECTABLE_NOISE_RATIO`
    (P2 — un DBSCAN qui ne rattache que quelques points très compacts peut
    afficher une silhouette artificiellement haute, calculée uniquement sur
    ces points-là). Les candidats disqualifiés restent dans la liste
    retournée (transparence du leaderboard), simplement relégués après ceux
    qui respectent le budget. Fonction pure — extraite pour être testée sans
    recalculer un vrai clustering.

    Retourne (liste_classée, noise_budget_exceeded_for_all) — le second
    élément est `True` seulement quand AUCUN candidat ne respecte le budget,
    auquel cas on retombe sur le classement brut plutôt que de ne renvoyer
    aucun résultat."""
    selectable = [c for c in valid if c["noise_ratio"] <= MAX_SELECTABLE_NOISE_RATIO]
    excluded = [c for c in valid if c["noise_ratio"] > MAX_SELECTABLE_NOISE_RATIO]
    excluded.sort(key=lambda c: c["silhouette"], reverse=True)
    if selectable:
        _attach_composite_rank(selectable)
        # Égalité de rang composite (rare, ex. 2 candidats parfaitement
        # complémentaires) : silhouette comme départage — seule des 3
        # métriques bornée et directement interprétable, même raisonnement
        # que le commentaire historique sur le choix de la silhouette.
        selectable.sort(key=lambda c: (c["composite_rank"], -c["silhouette"]))
        return selectable + excluded, False
    return excluded, True


def _compute_cluster_stability(
    X_processed: np.ndarray, spec: Any, params: dict[str, Any], seed: int
) -> Optional[float]:
    """Stabilité de la configuration gagnante par sous-échantillonnage — voir
    la constante `N_STABILITY_ROUNDS` ci-dessus pour le raisonnement complet.
    `None` (dégradation honnête, jamais un score inventé) si trop peu de
    points pour une estimation fiable, ou si moins de 2 sous-échantillons ont
    pu être ajustés avec succès."""
    n = X_processed.shape[0]
    if n < MIN_ROWS_FOR_STABILITY:
        return None

    working = X_processed if n <= MAX_ROWS_FOR_STABILITY else X_processed[:MAX_ROWS_FOR_STABILITY]
    n_working = working.shape[0]
    subsample_size = max(10, int(n_working * STABILITY_SUBSAMPLE_FRACTION))
    rng = np.random.default_rng(seed)

    runs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(N_STABILITY_ROUNDS):
        idx = rng.choice(n_working, size=subsample_size, replace=False)
        try:
            estimator = spec.build_estimator(dict(params), seed + i + 1)
            labels = np.asarray(estimator.fit_predict(working[idx]))
        except Exception:
            # Une configuration qui dégénère sur un sous-échantillon (ex.
            # trop peu de points distincts) est simplement ignorée — la
            # stabilité se calcule sur les rounds qui ont réussi, jamais un
            # plantage pour un indicateur secondaire.
            continue
        runs.append((idx, labels))

    if len(runs) < 2:
        return None

    scores: list[float] = []
    for (idx_a, labels_a), (idx_b, labels_b) in zip(runs, runs[1:]):
        pos_a = {int(v): k for k, v in enumerate(idx_a)}
        pos_b = {int(v): k for k, v in enumerate(idx_b)}
        common = sorted(set(pos_a) & set(pos_b))
        if len(common) < 5:
            continue
        la = [labels_a[pos_a[c]] for c in common]
        lb = [labels_b[pos_b[c]] for c in common]
        scores.append(float(adjusted_rand_score(la, lb)))

    return float(np.mean(scores)) if scores else None


def train_and_evaluate_clustering(
    X: pd.DataFrame,
    config: ClusteringConfig,
    progress_cb: ProgressCallback,
) -> ClusteringResult:
    """Point d'entrée principal — compare plusieurs configurations
    (algorithme × hyperparamètres) du registre, sélectionne la meilleure sur
    le score de silhouette, calcule les profils de segments du candidat
    retenu. Jamais de `y` : le clustering n'a pas de cible, contrairement au
    supervisé (`ml_training.py::train_and_evaluate`)."""
    progress_cb("Préparation des données", 5)
    n_samples_total = int(len(X))

    # Échantillonnage déterministe si le dataset dépasse MAX_ROWS_FOR_CLUSTERING
    # — même pattern que `dimensionality_training.py`/`anomaly_training.py`
    # (index d'origine préservé via une colonne technique, retirée avant le
    # préprocesseur). Sans ce plafond, le clustering hiérarchique (O(n²) en
    # mémoire) pouvait déclencher un MemoryError en cours de job sur un gros
    # dataset, sans avertissement préalable à l'utilisateur.
    X_indexed = X.reset_index(drop=True).copy()
    X_indexed[_ROW_INDEX_COLUMN] = np.arange(n_samples_total)
    row_cap = _effective_row_cap(config.algorithm_ids)
    sampled_flag = n_samples_total > row_cap
    X_sampled = sample_if_large(X_indexed, row_cap, config.seed)
    X_used = X_sampled.drop(columns=[_ROW_INDEX_COLUMN]).reset_index(drop=True)
    n_samples_used = int(len(X_used))

    preprocessor = build_preprocessor(X_used)
    X_processed = preprocessor.fit_transform(X_used)
    if hasattr(X_processed, "toarray"):
        # Contrairement au supervisé (Lot 3, sparse préservé pour les gros
        # volumes), tous les algorithmes de ce registre (KMeans, DBSCAN,
        # hiérarchique) exigent une entrée dense — un usage exploratoire de
        # clustering porte typiquement sur des jeux de données plus modestes
        # que l'entraînement supervisé, le risque mémoire est bien moindre
        # une fois le plafond ci-dessus appliqué.
        X_processed = X_processed.toarray()

    specs = specs_for(config.algorithm_ids)
    if not specs:
        # Garde-fou défensif seulement — l'API valide déjà les ids en amont
        # (même pattern que `ml_training.py`), ne devrait jamais arriver.
        specs = specs_for(None)

    all_configs = [
        (spec, cand)
        for spec in specs
        for cand in spec.candidate_configs(X_processed.shape[0], config.seed)
    ]
    if not all_configs:
        raise TrainingAbortedError(
            "Aucune configuration de clustering n'est applicable à ce jeu de données "
            "(trop peu de lignes). Essayez avec un jeu de données plus grand."
        )

    progress_cb(f"Évaluation de {len(all_configs)} configurations", 15)

    evaluated: list[dict[str, Any]] = []
    total = len(all_configs)
    for i, (spec, cand) in enumerate(all_configs):
        params = dict(cand.params)
        if spec.id == "dbscan" and params.get("eps") is None:
            eps = resolve_dbscan_eps(X_processed, params["min_samples"])
            if eps <= 0:
                # Dégénéré (beaucoup de points strictement identiques après
                # préprocessing — DBSCAN rejette `eps<=0`) : candidat ignoré
                # plutôt qu'un plantage, les autres configurations du
                # catalogue restent évaluées normalement.
                progress_cb(
                    f"Évaluation de {len(all_configs)} configurations",
                    15 + int(70 * (i + 1) / total),
                )
                continue
            params["eps"] = eps

        estimator = spec.build_estimator(params, config.seed)
        labels = np.asarray(estimator.fit_predict(X_processed))
        metrics = _compute_cluster_metrics(X_processed, labels)

        evaluated.append(
            {
                "spec": spec,
                "cand_label": cand.label,
                "params": params,
                "labels": labels,
                "estimator": estimator,
                **metrics,
            }
        )
        progress_cb(
            f"Évaluation de {len(all_configs)} configurations",
            15 + int(70 * (i + 1) / total),
        )

    valid = [c for c in evaluated if c["silhouette"] is not None]
    if not valid:
        raise TrainingAbortedError(
            "Aucune configuration testée n'a produit de regroupement exploitable sur ce jeu "
            "de données (un seul groupe détecté, ou tous les points classés atypiques). "
            "Essayez avec d'autres variables, ou vérifiez qu'il existe une vraie structure "
            "de groupes dans vos données."
        )

    # Score de sélection = rang composite sur les 3 métriques (voir
    # `_attach_composite_rank`) — PLUS la silhouette seule depuis ce
    # correctif (retour utilisateur direct, observé deux fois : la
    # silhouette seule élisait une configuration nettement pire en
    # Davies-Bouldin pour un gain marginal de silhouette). Silhouette
    # gardée comme départage en cas d'égalité de rang composite (seule des
    # 3 métriques bornée [-1, 1] et directement interprétable).
    valid, noise_budget_exceeded_for_all = _rank_candidates_with_noise_budget(valid)

    progress_cb("Sélection du meilleur regroupement", 88)

    # Résultats complets (profils de segments) pour le TOP 3 seulement
    # (retour utilisateur direct : "propose les 3 meilleurs modèles,
    # résultats propres pour chaque, laisse le choix à l'utilisateur") —
    # jamais pour le reste du classement (coût de calcul + volume de
    # réponse non justifiés pour des configurations que l'utilisateur ne
    # consultera probablement jamais). `labels` déjà disponible pour
    # CHAQUE candidat évalué (calculé une seule fois plus haut, jamais
    # refit ici) — ce correctif ne coûte donc que le calcul des profils
    # eux-mêmes (statistiques descriptives), pas un nouveau clustering.
    TOP_N_WITH_FULL_RESULTS = 3

    candidates_result: list[ClusterCandidateResult] = []
    for rank, c in enumerate(valid, start=1):
        candidate_profiles = None
        candidate_noise_count = None
        if rank <= TOP_N_WITH_FULL_RESULTS:
            candidate_profiles = _build_cluster_profiles(X_used, c["labels"])
            candidate_noise_count = int((c["labels"] == -1).sum())
        candidates_result.append(
            ClusterCandidateResult(
                algorithm_id=c["spec"].id,
                label=c["cand_label"],
                family=c["spec"].family,
                params={k: v for k, v in c["params"].items()},
                n_clusters=c["n_clusters"],
                silhouette=c["silhouette"],
                davies_bouldin=c["davies_bouldin"],
                calinski_harabasz=c["calinski_harabasz"],
                noise_ratio=c["noise_ratio"],
                is_winner=(rank == 1),
                rank=rank,
                # `.get(...)` : `_attach_composite_rank` n'est appelé que sur
                # les candidats DANS le budget de bruit (`selectable`) —
                # un candidat exclu (`excluded`, hors budget) n'a jamais ces
                # clés, `None` est alors le bon signal (jamais classé).
                rank_silhouette=c.get("rank_silhouette"),
                rank_davies_bouldin=c.get("rank_davies_bouldin"),
                rank_calinski_harabasz=c.get("rank_calinski_harabasz"),
                composite_rank=c.get("composite_rank"),
                cluster_profiles=candidate_profiles,
                noise_count=candidate_noise_count,
            )
        )

    winner = valid[0]
    progress_cb("Calcul des profils de segments", 92)
    # Déjà calculé ci-dessus (rang 1 fait toujours partie du top 3) — jamais
    # recalculé une seconde fois.
    profiles = candidates_result[0].cluster_profiles or _build_cluster_profiles(X_used, winner["labels"])

    progress_cb("Vérification de la stabilité", 96)
    stability_ari = _compute_cluster_stability(X_processed, winner["spec"], winner["params"], config.seed)

    # Données d'assignation de nouvelles observations (Lot 6B, §F.2 —
    # jusqu'ici, un clustering entraîné ne pouvait jamais être réutilisé).
    # KMeans/MiniBatchKMeans exposent déjà `.predict()` en natif (rien à
    # ajouter). Hiérarchique/DBSCAN n'ont PAS de `.predict()` en sklearn
    # (modèles transductifs) — voir `services/clustering_inference.py` pour
    # les approximations retenues, calculées ici une seule fois (jamais
    # recalculées à l'inférence) et persistées dans le pipeline_bundle.
    assignment_bundle_extra: dict[str, Any] = {}
    if winner["spec"].id == "hierarchical":
        labels_arr = winner["labels"]
        assignment_bundle_extra["centroids"] = {
            int(cid): X_processed[labels_arr == cid].mean(axis=0) for cid in sorted(set(labels_arr.tolist()))
        }
    elif winner["spec"].id == "dbscan":
        core_idx = winner["estimator"].core_sample_indices_
        assignment_bundle_extra["core_points"] = X_processed[core_idx]
        assignment_bundle_extra["core_labels"] = winner["labels"][core_idx]
        assignment_bundle_extra["eps"] = winner["params"]["eps"]

    model_card = {
        "algorithm": winner["spec"].label,
        "algorithm_id": winner["spec"].id,
        "family": winner["spec"].family,
        "n_clusters": winner["n_clusters"],
        # `n_samples` = nombre RÉELLEMENT clusterisé (X_used, potentiellement
        # échantillonné) — conservé sous ce nom pour compatibilité avec les
        # lecteurs existants (ex. `Clustering.tsx::totalSamples`, dont la
        # somme des tailles de profils doit rester égale à cette valeur).
        "n_samples": n_samples_used,
        "n_samples_total": n_samples_total,
        "n_samples_used": n_samples_used,
        "sampled": sampled_flag,
        # Transparence sur LE plafond réellement appliqué à ce job (voir
        # `_effective_row_cap`) — permet au frontend d'expliquer PLUTÔT que
        # "échantillonnage" en général, POURQUOI ce chiffre précis (registre
        # par défaut avec hiérarchique -> conservateur, ou partitionnement
        # seul -> plafond plus généreux).
        "row_cap": row_cap,
        "row_cap_linear_only": row_cap == MAX_ROWS_FOR_CLUSTERING_LINEAR_ONLY,
        "silhouette": winner["silhouette"],
        "davies_bouldin": winner["davies_bouldin"],
        "calinski_harabasz": winner["calinski_harabasz"],
        "noise_ratio": winner["noise_ratio"],
        "n_candidates_evaluated": len(all_configs),
        "seed": config.seed,
        # Transparence sur le garde-fou bruit (P2) — jamais silencieux quand
        # aucune configuration testée ne respecte le budget par défaut.
        "noise_budget_exceeded_for_all": noise_budget_exceeded_for_all,
        # `None` = pas assez de points pour une estimation fiable (dégradation
        # honnête) — jamais un score inventé (Lot 6B, §F.2).
        "stability_ari": stability_ari,
    }

    progress_cb("Terminé", 100)

    return ClusteringResult(
        winning_algorithm_id=winner["spec"].id,
        winning_label=winner["cand_label"],
        all_candidates=candidates_result,
        labels=[int(v) for v in winner["labels"]],
        noise_count=int((winner["labels"] == -1).sum()),
        cluster_profiles=profiles,
        feature_columns=list(X.columns),
        model_card=model_card,
        pipeline_bundle={
            "preprocessor": preprocessor,
            "model": winner["estimator"],
            "algorithm_id": winner["spec"].id,
            **assignment_bundle_extra,
        },
    )
