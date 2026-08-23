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

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
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
    # Thème d'interface choisi (Lot UI — refonte visuelle) : "graphite" | "ivoire"
    # | "minuit" | "ardoise" | "porcelaine". Server_default="graphite" pour que
    # la migration s'applique sans backfill sur les lignes existantes — c'est
    # aussi le thème par défaut du ThemeProvider frontend (ordre de résolution :
    # ce champ → localStorage → prefers-color-scheme → graphite).
    ui_theme: Mapped[str] = mapped_column(String(20), nullable=False, server_default="graphite")

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
    # Lot 5 (correctif P2, AUDIT_DATALAB_2026-08-16.md §P2) — SHA-256 du
    # contenu brut du fichier, calculé une seule fois à l'upload (voir
    # api/routers/datasets.py::upload_dataset). NULL pour l'historique
    # antérieur à ce lot (jamais recalculé a posteriori — recalculer
    # exigerait de relire chaque fichier sur disque, hors périmètre d'une
    # migration de schéma). Sert à détecter un ré-upload accidentel du
    # même fichier (voir duplicate_of_dataset_id) et, plus généralement,
    # à vérifier l'intégrité d'un fichier stocké.
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # Renseigné à l'upload si un dataset de LA MÊME organisation partage
    # déjà exactement le même content_hash — jamais bloquant (l'upload
    # aboutit toujours), purement informatif pour éviter un doublon
    # silencieux. ondelete SET NULL : si le dataset d'origine est
    # supprimé, ce dataset-ci reste un dataset normal, pas orphelin.
    duplicate_of_dataset_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
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
    # Lot 5 (correctif P1) — dénormalisé depuis `training_job.dataset_id` :
    # `target_column` (ci-dessous) l'est déjà pour la même raison (identifier
    # le "problème" — dataset + cible — sans jointure sur `training_jobs` à
    # chaque requête). Immuable après création, comme `training_job_id`.
    dataset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
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
    # Lot 9 — registre de modèles versionné. `stage` :
    # "staging"/"production"/"archived" (Lot 5, correctif P1, ce dernier
    # ajouté ici), NULL = jamais promu (comportement historique,
    # rétrocompat par absence comme le reste du projet — voir
    # api/routers/training.py::promote_model pour la règle "un seul
    # modèle en production par dataset+cible", et `_VALID_STAGES` pour la
    # liste faisant foi).
    stage: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    promoted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Lot 5 (correctif P1) — numéro de version au sein du "problème"
    # (organization_id, dataset_id, target_column) : 1 pour le tout
    # premier modèle entraîné sur ce couple dataset/cible, incrémenté à
    # chaque nouveau (voir services/model_versioning.py::next_version).
    # Assigné UNE SEULE FOIS à la création, jamais recalculé — un modèle
    # "version 3" garde ce numéro pour toujours, y compris si une version
    # intermédiaire est supprimée par la suite (l'historique reste
    # interprétable, jamais de renumérotation rétroactive).
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        # Empêche deux modèles du même problème de partager le même numéro
        # de version — filet de sécurité en cas de course entre deux
        # entraînements terminant au même instant sur le même dataset+cible
        # (deux workers RQ concurrents, voir services/model_versioning.py).
        UniqueConstraint("organization_id", "dataset_id", "target_column", "version", name="uq_ml_models_version"),
    )

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


class Prediction(Base):
    """Une prédiction individuelle produite par
    `POST /training/jobs/{id}/predict` (Lot 5, correctif I2,
    AUDIT_DATALAB_2026-08-16.md §I2).

    Avant ce lot, chaque prédiction disparaissait sitôt la réponse HTTP
    envoyée : aucune trace de CE qui a été demandé, CE qui a été répondu,
    ni PAR QUI — un incident ("le modèle a mal prédit pour ce dossier")
    ne pouvait pas être investigué après coup. Remonte au modèle (donc au
    run d'entraînement, au dataset, à l'organisation) via `ml_model_id` —
    jamais dupliqué ici.

    `output_json` capture prédiction + probabilités + intervalle CQR
    (tout ce que l'audit désigne par "sortie" et "intervalle") — jamais
    `explanation` (SHAP local) : recalculable à la demande depuis le
    bundle + `input_json`, volumineuse, pas une donnée qui fait foi (voir
    services/ml_inference.py::explain_one, appelé sans persistance)."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ml_model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False, index=True
    )
    requested_by_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    input_json: Mapped[str] = mapped_column(Text, nullable=False)
    output_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    ml_model: Mapped["MLModel"] = relationship("MLModel")
    requested_by: Mapped[Optional["User"]] = relationship("User")


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


class DimensionalityJob(Base):
    """Un calcul de réduction de dimension (Lot 13, ML non supervisé) —
    même mécanisme de tâche de fond que `ClusteringJob` (RQ, `training_queue`
    partagée). Table DÉDIÉE, même raisonnement que `ClusteringJob` :
    conventions d'isolation/de progression identiques, mais pas de notion de
    cible ni de comparaison multi-algorithmes (une seule méthode choisie par
    job, voir services/dimensionality_registry.py)."""

    __tablename__ = "dimensionality_jobs"

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
    feature_columns_json: Mapped[str] = mapped_column(Text, nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)  # {"algorithm_id":, "seed":}
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
    result: Mapped[Optional["DimensionalityModel"]] = relationship(
        "DimensionalityModel", back_populates="dimensionality_job", uselist=False, passive_deletes=True
    )


class DimensionalityModel(Base):
    """Le résultat produit par un `DimensionalityJob` réussi — méthode
    retenue, variance expliquée/loadings (PCA, toujours calculée en plus de
    la méthode choisie), mesures de fidélité (`trustworthiness`), note
    obligatoire sur la fidélité des distances."""

    __tablename__ = "dimensionality_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dimensionality_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dimensionality_jobs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    algorithm_id: Mapped[str] = mapped_column(String(30), nullable=False)
    n_samples_total: Mapped[int] = mapped_column(Integer, nullable=False)
    n_samples_used: Mapped[int] = mapped_column(Integer, nullable=False)
    sampled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    trustworthiness_primary: Mapped[float] = mapped_column(Float, nullable=False)
    trustworthiness_pca: Mapped[float] = mapped_column(Float, nullable=False)
    variance_explained_json: Mapped[str] = mapped_column(Text, nullable=False)  # [ratio_pc1, ratio_pc2]
    total_variance_explained: Mapped[float] = mapped_column(Float, nullable=False)
    loadings_json: Mapped[str] = mapped_column(Text, nullable=False)  # [{"feature":, "pc1":, "pc2":}, ...]
    distance_fidelity_note: Mapped[str] = mapped_column(Text, nullable=False)
    feature_columns_json: Mapped[str] = mapped_column(Text, nullable=False)
    model_card_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    dimensionality_job: Mapped["DimensionalityJob"] = relationship("DimensionalityJob", back_populates="result")


class DimensionalityPoint(Base):
    """Une observation projetée en 2D par un `DimensionalityJob` — une ligne
    par point (pas un JSON agrégé), même raisonnement que
    `ClusterCandidateRecord` : requêtable, borné par l'échantillonnage
    (`MAX_ROWS_FOR_EMBEDDING`, services/dimensionality_training.py). Pas de
    `created_at` — table à forte cardinalité (jusqu'à 5000 lignes/job)
    insérée en une seule transaction, un horodatage par ligne n'apporterait
    aucune information supplémentaire."""

    __tablename__ = "dimensionality_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dimensionality_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dimensionality_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)  # position dans le dataset d'origine
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)


class AnomalyJob(Base):
    """Une détection d'anomalies (Lot 14, ML non supervisé) — même
    mécanisme de tâche de fond que `ClusteringJob`/`DimensionalityJob` (RQ,
    `training_queue` partagée). Table DÉDIÉE, mêmes conventions
    d'isolation/de progression. Pas de notion d'algorithme choisi par
    l'utilisateur (`config_json` ne porte que `top_n`/`seed`) — Isolation
    Forest et LOF tournent systématiquement ensemble (voir
    services/anomaly_registry.py)."""

    __tablename__ = "anomaly_jobs"

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
    feature_columns_json: Mapped[str] = mapped_column(Text, nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)  # {"top_n":, "seed":}
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
    result: Mapped[Optional["AnomalyModel"]] = relationship(
        "AnomalyModel", back_populates="anomaly_job", uselist=False, passive_deletes=True
    )


class AnomalyModel(Base):
    """Le résultat produit par un `AnomalyJob` réussi — comptages/taux par
    algorithme et en consensus, histogramme des scores de consensus (même
    forme que `compute_histogram`, services/dataset_eda.py)."""

    __tablename__ = "anomaly_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    anomaly_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("anomaly_jobs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    n_samples_total: Mapped[int] = mapped_column(Integer, nullable=False)
    n_samples_used: Mapped[int] = mapped_column(Integer, nullable=False)
    sampled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    n_anomalies_isolation_forest: Mapped[int] = mapped_column(Integer, nullable=False)
    n_anomalies_lof: Mapped[int] = mapped_column(Integer, nullable=False)
    n_anomalies_consensus: Mapped[int] = mapped_column(Integer, nullable=False)
    anomaly_rate_isolation_forest: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_rate_lof: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_rate_consensus: Mapped[float] = mapped_column(Float, nullable=False)
    score_histogram_json: Mapped[str] = mapped_column(Text, nullable=False)  # {"bin_edges":, "counts":}
    feature_columns_json: Mapped[str] = mapped_column(Text, nullable=False)
    model_card_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    anomaly_job: Mapped["AnomalyJob"] = relationship("AnomalyJob", back_populates="result")


class AnomalyObservationRecord(Base):
    """Une observation classée parmi les plus atypiques d'un `AnomalyJob` —
    seul le top-N (borné, `MAX_TOP_N`) est persisté, une ligne par
    observation, même raisonnement que `ClusterCandidateRecord`/
    `DimensionalityPoint` : requêtable, jamais un JSON agrégé."""

    __tablename__ = "anomaly_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    anomaly_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("anomaly_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    consensus_score: Mapped[float] = mapped_column(Float, nullable=False)
    score_isolation_forest: Mapped[float] = mapped_column(Float, nullable=False)
    score_lof: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly_isolation_forest: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_anomaly_lof: Mapped[bool] = mapped_column(Boolean, nullable=False)
    agreement: Mapped[str] = mapped_column(String(30), nullable=False)
    numeric_deviations_json: Mapped[str] = mapped_column(Text, nullable=False)
    categorical_flags_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class VisionDataset(Base):
    """Un dataset d'images uploadé (pilier Vision, Lot 15 sous-lot A) —
    fondation partagée par la classification d'images et la détection
    d'anomalies visuelles (structure_type distingue les deux, valeur "mvtec_ad" pour la structure normal/défaut).
    Contrairement à `Dataset` (un seul fichier tabulaire), un dataset vision
    est un dossier d'images extraites du ZIP uploadé — voir
    `api/core/storage.py::vision_dataset_dir`."""

    __tablename__ = "vision_datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    uploaded_by_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    structure_type: Mapped[str] = mapped_column(String(20), nullable=False)  # classification | mvtec_ad
    storage_dir: Mapped[str] = mapped_column(String(500), nullable=False)
    n_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_classes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    class_distribution_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    validation_report_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="processing")  # processing | ready | error
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    uploaded_by: Mapped[Optional["User"]] = relationship("User")


class VisionClassificationJob(Base):
    """Un entraînement de classification d'images par transfer learning
    (pilier Vision, Lot 15 sous-lot B) — même mécanisme de tâche de fond que
    `AnomalyJob`/`ClusteringJob` (RQ, `training_queue` partagée). Le dataset
    source doit être un `VisionDataset` de structure "classification"
    (vérifié par le router avant l'enfilage)."""

    __tablename__ = "vision_classification_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    vision_dataset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("vision_datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_by_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # {"backbone_id":, "num_epochs":, "batch_size":, "learning_rate":,
    #  "dropout_rate":, "freeze_backbone":, "unfreeze_after_epoch":, "seed":}
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
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
    vision_dataset: Mapped["VisionDataset"] = relationship("VisionDataset")
    created_by: Mapped[Optional["User"]] = relationship("User")
    result: Mapped[Optional["VisionClassificationModel"]] = relationship(
        "VisionClassificationModel", back_populates="job", uselist=False, passive_deletes=True
    )


class VisionClassificationModel(Base):
    """Le résultat produit par un `VisionClassificationJob` réussi —
    métriques de test, historique d'entraînement (courbes train/val) et
    exemples de prédictions (corrects ET erronés, skill Computer Vision)."""

    __tablename__ = "vision_classification_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    vision_classification_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("vision_classification_jobs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    backbone_id: Mapped[str] = mapped_column(String(50), nullable=False)
    class_names_json: Mapped[str] = mapped_column(Text, nullable=False)
    n_train: Mapped[int] = mapped_column(Integer, nullable=False)
    n_val: Mapped[int] = mapped_column(Integer, nullable=False)
    n_test: Mapped[int] = mapped_column(Integer, nullable=False)
    history_json: Mapped[str] = mapped_column(Text, nullable=False)  # liste de métriques par époque
    test_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    test_precision_macro: Mapped[float] = mapped_column(Float, nullable=False)
    test_recall_macro: Mapped[float] = mapped_column(Float, nullable=False)
    test_f1_macro: Mapped[float] = mapped_column(Float, nullable=False)
    confusion_matrix_json: Mapped[str] = mapped_column(Text, nullable=False)
    examples_json: Mapped[str] = mapped_column(Text, nullable=False)
    model_card_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Lot 6A (correctif 16G, AUDIT_DATALAB_2026-08-16.md §16G) — NULL sur les
    # modèles entraînés avant ce correctif (rétrocompatibilité par absence,
    # même motif que model_card_json ci-dessus) : jamais un dict vide qui
    # suggérerait un calcul fait mais nul.
    roc_curves_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pr_curves_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    test_roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    job: Mapped["VisionClassificationJob"] = relationship("VisionClassificationJob", back_populates="result")


class VisionAnomalyJob(Base):
    """Un entraînement de détection d'anomalies visuelles (structure normal/défaut, pilier
    Vision, Lot 15 sous-lot C) — même mécanisme de tâche de fond que
    `VisionClassificationJob`. Le dataset source doit être un
    `VisionDataset` de structure "mvtec_ad" (vérifié par le router avant
    l'enfilage, revérifié par le worker)."""

    __tablename__ = "vision_anomaly_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    vision_dataset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("vision_datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_by_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # {"model_id":, "num_epochs":, "batch_size":, "learning_rate":, "seed":, "mask_percentile":}
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
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
    vision_dataset: Mapped["VisionDataset"] = relationship("VisionDataset")
    created_by: Mapped[Optional["User"]] = relationship("User")
    result: Mapped[Optional["VisionAnomalyModel"]] = relationship(
        "VisionAnomalyModel", back_populates="job", uselist=False, passive_deletes=True
    )


class VisionAnomalyModel(Base):
    """Le résultat produit par un `VisionAnomalyJob` réussi — seuil calibré
    (J de Youden sur `test/`, correctif du bug #7/#12 : plus un percentile
    arbitraire sur le train), métriques de détection, historique
    d'entraînement (reconstruction uniquement, pas de notion d'accuracy
    pendant l'entraînement — c'est un autoencodeur)."""

    __tablename__ = "vision_anomaly_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    vision_anomaly_job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("vision_anomaly_jobs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    model_id: Mapped[str] = mapped_column(String(50), nullable=False)
    n_train: Mapped[int] = mapped_column(Integer, nullable=False)
    n_val: Mapped[int] = mapped_column(Integer, nullable=False)
    n_test: Mapped[int] = mapped_column(Integer, nullable=False)
    # Correctif C2 (AUDIT_DATALAB_2026-08-16.md, Lot 0.2) — NULL sur les
    # modèles entraînés avant ce correctif (rétrocompatibilité par absence,
    # même motif que les colonnes ci-dessous) : n_test seul ne dit pas si le
    # seuil a été calibré sur un sous-ensemble distinct de celui évalué.
    n_calibration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    n_evaluation: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    history_json: Mapped[str] = mapped_column(Text, nullable=False)  # liste {epoch, train_loss, val_loss}
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    roc_auc: Mapped[float] = mapped_column(Float, nullable=False)
    test_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    test_precision: Mapped[float] = mapped_column(Float, nullable=False)
    test_recall: Mapped[float] = mapped_column(Float, nullable=False)
    test_f1: Mapped[float] = mapped_column(Float, nullable=False)
    confusion_matrix_json: Mapped[str] = mapped_column(Text, nullable=False)
    model_card_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    organization: Mapped["Organization"] = relationship("Organization")
    job: Mapped["VisionAnomalyJob"] = relationship("VisionAnomalyJob", back_populates="result")
    examples: Mapped[list["VisionAnomalyExampleRecord"]] = relationship(
        "VisionAnomalyExampleRecord", back_populates="model", passive_deletes=True
    )


class VisionAnomalyExampleRecord(Base):
    """Une image d'exemple classée par score d'anomalie décroissant (top-N
    borné, `MAX_EXAMPLES`) — heatmap ET masque binaire déjà encodés en PNG
    base64 (`services/vision_localization.py`), directement affichables,
    déjà réalignés à la taille de l'image originale (correctif des bugs
    #10/#17). Table dédiée plutôt qu'un JSON agrégé sur `VisionAnomalyModel`
    — même raisonnement que `AnomalyObservationRecord`."""

    __tablename__ = "vision_anomaly_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    vision_anomaly_model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("vision_anomaly_models.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relative_path: Mapped[str] = mapped_column(String(500), nullable=False)
    defect_category: Mapped[str] = mapped_column(String(100), nullable=False)
    true_label: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_label: Mapped[int] = mapped_column(Integer, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    heatmap_png: Mapped[str] = mapped_column(Text, nullable=False)
    mask_png: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Bug réel trouvé (Lot 10, exposé par l'ajout de `Feedback.author` — voir
    # JOURNAL.md) : `VisionAnomalyModel.examples` déclare `back_populates=
    # "model"` depuis toujours, mais ce côté de la relation ne l'avait jamais
    # défini — SQLAlchemy ne configure les mappers qu'à la première requête
    # ORM du process, jamais détecté avant. Corrigé en ajoutant la relation
    # manquante, jamais un simple contournement du symptôme.
    model: Mapped["VisionAnomalyModel"] = relationship("VisionAnomalyModel", back_populates="examples")


class Feedback(Base):
    """Retour utilisateur libre depuis le centre d'aide (Lot 10, refonte UI —
    retour direct : « ajoute un formulaire pour renseigner ce problème »
    plutôt qu'un simple lien mailto vers un support qui n'existe pas).
    Stocké tel quel, jamais traité automatiquement : consultable par les
    administrateurs de l'organisation concernée, jamais visible d'une autre
    (même isolation par organization_id que le reste de l'app)."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    # Chemin de la page depuis laquelle le retour a été envoyé (ex. "/aide",
    # "/training?job=45") — contexte utile pour comprendre le signalement,
    # jamais interprété/parsé côté serveur.
    page: Mapped[str] = mapped_column(String(300), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    author: Mapped["User"] = relationship("User")
