"""Modèles ORM — Organisation (bureau d'études) et utilisateurs.

Modèle multi-tenant retenu (voir le diagnostic de migration, section E) :
une Organisation = un bureau d'études, plusieurs utilisateurs y appartiennent
(rôle "owner" ou "member"). Toute donnée métier créée dans les lots suivants
(datasets, jobs d'entraînement, modèles) portera un organization_id et ne
sera jamais visible en dehors de son organisation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.core.database import Base


class Organization(Base):
    """Un bureau d'études — unité d'isolation des données."""

    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    users: Mapped[list["User"]] = relationship("User", back_populates="organization")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    nom: Mapped[str] = mapped_column(String(100), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    # "owner" : a créé l'organisation, seul rôle autorisé à inviter des membres (Lot 1).
    # "member" : rattaché à une organisation existante par un owner.
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="member")
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    actif: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")

    @property
    def is_owner(self) -> bool:
        return self.role == "owner"

    @property
    def organization_name(self) -> str:
        """Exposé pour les réponses API — évite de sérialiser l'objet Organization entier."""
        return self.organization.name


class Dataset(Base):
    """Un jeu de données tabulaire uploadé — appartient à l'organisation entière
    (pas seulement à qui l'a uploadé), cohérent avec le principe d'équipe partagée."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    uploaded_by_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    column_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # JSON list [{"name": ..., "dtype": ...}, ...] — évite une table séparée pour Lot 2
    columns_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="processing")  # processing | ready | error
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    uploaded_by: Mapped[Optional["User"]] = relationship("User")


class TrainingJob(Base):
    """Un entraînement ML lancé sur un dataset — exécuté en tâche de fond (RQ).

    `progress_step`/`progress_percent` sont mis à jour par le worker pendant
    l'exécution (voir workers/training_worker.py) ; le frontend les lit par
    polling (`GET /training/jobs/{id}`).
    """

    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_by_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    task_type: Mapped[str] = mapped_column(String(20), nullable=False)  # classification | regression
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_columns_json: Mapped[str] = mapped_column(Text, nullable=False)
    group_column: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    # Spec de feature engineering approuvée par l'utilisateur (Lot 4c),
    # {"version": ..., "upstream": [...], "pipeline": {...}} — voir
    # services/feature_engineering.py. Absente/NULL : comportement inchangé.
    feature_engineering_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # queued | running | completed | failed
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True)
    progress_step: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    progress_percent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Dernier signal de vie du worker (mis à jour à chaque étape de
    # progression, voir workers/training_worker.py::_make_progress_callback)
    # — permet à services/job_watchdog.py de distinguer un job réellement en
    # cours d'un job "running" abandonné par un worker mort.
    progress_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rq_job_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    dataset: Mapped["Dataset"] = relationship("Dataset")
    created_by: Mapped[Optional["User"]] = relationship("User")
    # passive_deletes=True : ne pas mettre training_job_id à NULL en Python à
    # la suppression (impossible, colonne NOT NULL) — laisser le ON DELETE
    # CASCADE de la contrainte FK (voir MLModel.training_job_id) faire le
    # travail côté base de données.
    model: Mapped[Optional["MLModel"]] = relationship(
        "MLModel", back_populates="training_job", uselist=False, passive_deletes=True
    )


class MLModel(Base):
    """Le modèle produit par un TrainingJob réussi — métriques, explicabilité
    SHAP et intervalles conformes (CQR) inclus, pas seulement l'artefact.

    Nommé `MLModel` (pas `Model`) pour ne jamais entrer en collision avec
    `sqlalchemy.orm.Mapped`/`pydantic.BaseModel` dans les imports.
    """

    __tablename__ = "ml_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    training_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)  # libellé lisible du registre (services/ml_registry.py, Lot 5)
    task_type: Mapped[str] = mapped_column(String(20), nullable=False)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_columns_json: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON [{"name": ..., "dtype": ...}, ...] des colonnes d'entrée (pas la
    # cible) — permet au frontend de générer un formulaire de prédiction
    # adapté (nombre vs texte) sans redemander le dataset d'origine (Lot 4).
    feature_schema_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    shap_summary_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cqr_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_card_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Matrice de confusion + courbes ROC/PR (classification) ou
    # prédit-vs-réel + résidus (régression) — pour les graphiques
    # d'évaluation (Lot 4b), pas pour les métriques brutes déjà dans metrics_json.
    evaluation_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Copie de la spec de feature engineering appliquée à ce modèle (Lot 4c),
    # pour affichage en toute transparence dans le résultat — la copie qui
    # fait foi pour le rejeu à l'inférence est celle du bundle joblib, pas
    # celle-ci (voir services/ml_inference.py).
    feature_engineering_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Explicabilité globale enrichie (Lot Explicabilité globale) — dict
    # clé=classe ("global" en régression/binaire) → points {feature,
    # feature_value, shap_value}, réutilisés depuis le même calcul SHAP que
    # shap_summary_json (pas un second appel à l'explainer).
    shap_beeswarm_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Importance par permutation — mesure alternative au SHAP, indépendante
    # du type de modèle, pour recouper shap_summary_json.
    permutation_importance_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Courbe de calibration (reliability diagram) — classification
    # uniquement, NULL en régression (non applicable, pas dégradé).
    calibration_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Courbe d'apprentissage (train-size vs score) — diagnostic de
    # sur/sous-apprentissage complémentaire à delta_r2/accuracy train-test.
    learning_curve_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Lot 9 — registre de modèles versionné. `stage` : "staging"/"production",
    # NULL = jamais promu (comportement historique, rétrocompat par absence
    # comme le reste du projet — voir api/routers/training.py::promote_model
    # pour la règle "un seul modèle en production par dataset+cible").
    stage: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    promoted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    training_job: Mapped["TrainingJob"] = relationship("TrainingJob", back_populates="model")


class ModelCandidate(Base):
    """Un modèle comparé pendant un TrainingJob — TOUS les candidats du
    catalogue par défaut, pas seulement le gagnant (Lot D, leaderboard).

    Avant ce lot, seul le gagnant était persisté (`MLModel`) : le travail de
    comparaison réel du moteur (`services/ml_training.py::train_and_evaluate`)
    restait invisible. Table dédiée plutôt qu'un JSON sur `TrainingJob` — le
    tri/filtre inter-jobs prévu (lot D-bis) a besoin de colonnes requêtables,
    pas d'un blob à désérialiser à chaque comparaison.

    N'existe QUE pour les jobs entraînés depuis ce lot : aucune ligne pour
    l'historique antérieur (rétrocompatibilité par absence, pas par
    backfill — voir `api/routers/training.py::get_job_candidates`)."""

    __tablename__ = "model_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    training_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)  # spec.label(task_type), registre (ml_registry.py)
    family: Mapped[str] = mapped_column(String(30), nullable=False)  # ModelSpec.family — pour grouper à l'affichage
    # LA métrique qui a réellement départagé les candidats (ROC-AUC pondérée
    # en classification, R² en régression — voir
    # services/ml_training.py::_classification_selection_score et
    # `scoring="r2"`) — jamais l'accuracy brute, trompeuse sur un dataset
    # déséquilibré (voir la correction de `_headline_metric`, Lot D 1/4).
    selection_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_winner: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)  # 1 = gagnant, pré-trié côté moteur
    # Score de sélection par fold de validation croisée — variance
    # inter-folds (option A du cadrage Lot D), capturée pendant la recherche
    # Optuna déjà en cours (aucun ré-entraînement, voir
    # `services/ml_training.py::_optimize_one_model`). NULL pour l'historique
    # antérieur à ce lot (jamais recalculé a posteriori).
    fold_scores_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Erreur en unité réelle de la cible (RMSE, validation croisée) —
    # régression uniquement : le R² seul n'est pas lisible pour un BE. NULL
    # en classification (pas de "secondaire" pertinent au même sens).
    secondary_metric: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    secondary_metric_label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    organization: Mapped["Organization"] = relationship("Organization")
    training_job: Mapped["TrainingJob"] = relationship("TrainingJob")


class AuditLog(Base):
    """Journal des actions sensibles (Lot 10 — durcissement SaaS) : qui a
    fait quoi, quand — ajout/désactivation de membre, suppression de
    dataset/entraînement, promotion de modèle. Pas un log applicatif complet
    (déjà couvert par les logs serveur, `logging` standard) : uniquement les
    actions qu'un `owner` pourrait vouloir auditer après coup ("qui a
    supprimé ce dataset ?"), consultable depuis l'équipe (`GET /auth/team/
    audit-log`, réservé au owner comme le reste de la gestion d'équipe).

    `actor_id` NULLABLE avec `ondelete="SET NULL"` : l'entrée doit survivre
    à la suppression du compte de son auteur (traçabilité de l'action même
    si l'utilisateur qui l'a faite est parti) — jamais de cascade delete
    sur ce lien, contrairement au reste du modèle de données."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    actor_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # Espace de noms pointé "ressource.action" (ex. "dataset.deleted",
    # "training_job.deleted", "member.added", "model.promoted") — assez
    # structuré pour filtrer plus tard, assez simple pour rester un champ
    # texte plutôt qu'une table d'énumération séparée.
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    target_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    target_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Contexte additionnel en langage clair (ex. {"dataset_name": "x.csv"}) —
    # pour rester lisible même après suppression de la ressource elle-même
    # (le nom d'un dataset supprimé ne serait plus consultable autrement).
    details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    actor: Mapped[Optional["User"]] = relationship("User")


class ClusteringJob(Base):
    """Un entraînement de clustering (Lot 11+, ML non supervisé) — exécuté
    en tâche de fond (RQ), même mécanisme que `TrainingJob` (réutilise la
    même `training_queue`, voir `api/core/job_queue.py`).

    Table DÉDIÉE, jamais une extension de `TrainingJob` — confirmé par
    l'audit du 2026-08-14 (AUDIT_ROADMAP.md) : le clustering n'a pas de
    cible (`target_column`, NOT NULL sur `TrainingJob`) ni de notion de
    split train/test au même sens. Mêmes conventions d'isolation
    (`organization_id`), de progression (`progress_step`/`progress_percent`/
    `progress_updated_at`, réutilisé par `services/job_watchdog.py`) et de
    quota que le supervisé."""

    __tablename__ = "clustering_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_by_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # Colonnes soumises au clustering — pas de "cible" (non supervisé).
    feature_columns_json: Mapped[str] = mapped_column(Text, nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    # queued | running | completed | failed
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True)
    progress_step: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    progress_percent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rq_job_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    dataset: Mapped["Dataset"] = relationship("Dataset")
    created_by: Mapped[Optional["User"]] = relationship("User")
    result: Mapped[Optional["ClusterModel"]] = relationship(
        "ClusterModel", back_populates="clustering_job", uselist=False, passive_deletes=True
    )


class ClusterModel(Base):
    """Le résultat produit par un `ClusteringJob` réussi — algorithme
    retenu, métriques (silhouette/Davies-Bouldin/Calinski-Harabasz), profils
    de segments et artefact persistés, même esprit que `MLModel` côté
    supervisé."""

    __tablename__ = "cluster_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clustering_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("clustering_jobs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)  # libellé lisible (services/clustering_registry.py)
    n_clusters: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_columns_json: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)  # silhouette/davies_bouldin/calinski_harabasz/noise_ratio
    # Profils de segments (taille, moyenne/médiane par variable, variables
    # différenciantes) — voir services/clustering_training.py::ClusterProfile.
    # Un LLM éventuel ne reçoit QUE ces statistiques déjà calculées, jamais
    # laissé à inventer les caractéristiques d'un cluster (skill
    # senior-ai-saas-engineer, data-science.md).
    profiles_json: Mapped[str] = mapped_column(Text, nullable=False)
    # Cluster assigné par ligne du dataset (-1 = bruit DBSCAN) — base d'une
    # future visualisation (Lot 13, réduction de dimension : coloration par
    # cluster) et de la prédiction du cluster d'une nouvelle observation.
    labels_json: Mapped[str] = mapped_column(Text, nullable=False)
    model_card_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    clustering_job: Mapped["ClusteringJob"] = relationship("ClusteringJob", back_populates="result")


class ClusterCandidateRecord(Base):
    """Un candidat comparé pendant un `ClusteringJob` — TOUS les candidats
    évalués (plusieurs k, plusieurs algorithmes), pas seulement le gagnant.
    Même raisonnement que `ModelCandidate` (Lot D, leaderboard supervisé) :
    rendre visible le travail de comparaison réel du moteur."""

    __tablename__ = "cluster_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clustering_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("clustering_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    family: Mapped[str] = mapped_column(String(30), nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    n_clusters: Mapped[int] = mapped_column(Integer, nullable=False)
    silhouette: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    davies_bouldin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calinski_harabasz: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    noise_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_winner: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    organization: Mapped["Organization"] = relationship("Organization")
    clustering_job: Mapped["ClusteringJob"] = relationship("ClusteringJob")
