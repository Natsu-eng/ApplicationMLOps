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
