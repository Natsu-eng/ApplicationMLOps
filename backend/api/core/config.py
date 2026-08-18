"""Configuration centralisée de l'API.

Toutes les valeurs sont surchargeables par variable d'environnement ou par le
fichier `backend/.env` (jamais commité — voir `backend/.env.example`).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env — deux niveaux au-dessus de ce fichier (backend/api/core/config.py)
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """Paramètres applicatifs. Un seul point de vérité pour toute la config backend."""

    # Identité de l'application (affichée dans /api/health et la doc Swagger)
    app_name: str = "DataLab Pro"
    app_version: str = "0.1.0"
    environment: str = "development"  # development | production

    # Réseau — utilisé pour le CORS (voir api/main.py)
    frontend_url: str = "http://localhost:5173"

    # Base de données : Postgres en production, SQLite en dev si DATABASE_URL absent
    # (même convention que CIAM — voir backend/api/core/database.py)
    database_url: str = "sqlite:///./database/datalab.db"

    # Authentification — JWT HS256 (voir api/core/security.py)
    # ⚠️ OBLIGATOIRE à changer en production. Générer :
    #    python -c "import secrets; print(secrets.token_hex(64))"
    jwt_secret_key: str = "changez-cette-cle-en-production"

    # Upload de datasets (Lot 2) — upload synchrone en mémoire, pas encore de
    # tâche de fond (Lot 3) : garder une limite raisonnable pour rester robuste.
    max_upload_size_mb: int = 200

    # File de tâches d'entraînement (Lot 3) — RQ + Redis, décision du diagnostic
    # de migration (CIAM n'en a pas besoin, ses tâches sont trop courtes ; un
    # entraînement avec recherche d'hyperparamètres Optuna ne l'est pas).
    redis_url: str = "redis://localhost:6379/0"

    # Entraînement ML (Lot 3) — défauts appliqués si le job n'en précise pas.
    optuna_trials_default: int = 20
    cv_folds_default: int = 4
    shap_sample_size: int = 500
    cqr_alpha: float = 0.20  # intervalle de confiance conforme à 80 % (1 - alpha)
    cqr_n_strata: int = 5  # strates Mondrian — voir services/ml_training.py
    model_seed: int = 42

    # Durcissement SaaS (Lot 10) — garde-fou technique, pas un plan
    # tarifaire : un seul worker RQ traite les jobs de TOUTES les
    # organisations (voir docker-compose.yml) ; sans limite, une
    # organisation qui enfile beaucoup d'entraînements d'affilée peut
    # affamer les autres. Valeur de départ prudente, pensée pour un usage BE
    # normal (quelques entraînements en parallèle au plus) — à revoir si un
    # vrai modèle de plans/quotas commerciaux est décidé (hors périmètre
    # technique de ce lot).
    max_concurrent_jobs_per_org: int = 3

    # Durcissement SaaS (H2, AUDIT_ROADMAP.md) — au-delà de ce délai sans
    # signal de vie (voir services/job_watchdog.py), un job "running" est
    # considéré orphelin (worker mort) et reclassé "failed". Volontairement
    # au-dessus du timeout RQ (`training_queue`, 1800s = 30 min,
    # api/core/job_queue.py) : ne doit jamais se déclencher sur un
    # entraînement encore réellement actif, seulement sur un worker disparu.
    stale_job_timeout_minutes: int = 40

    # Upload de datasets d'images (pilier Vision, Lot 15 sous-lot A) — ZIP
    # décompressé en mémoire côté service (services/vision_datasets.py) : un
    # dataset MVTec AD réel peut peser plusieurs centaines de Mo (contrairement
    # aux CSV tabulaires, d'où une limite dédiée plus haute que
    # max_upload_size_mb). max_vision_dataset_images protège en plus contre les
    # ZIP bombes (nombre d'entrées, indépendant de la taille compressée).
    max_vision_upload_size_mb: int = 500
    max_vision_dataset_images: int = 5000

    # Durcissement SaaS (H11, AUDIT_ROADMAP.md) — aucune limite n'existait
    # sur les tentatives de connexion échouées, brute force possible sans
    # borne. Fenêtre glissante par IP cliente, stockée dans Redis (déjà une
    # dépendance du projet, voir job_queue.py) — échec ouvert (log seul) si
    # Redis est indisponible, jamais bloquant pour la connexion légitime.
    login_rate_limit_max_attempts: int = 10
    login_rate_limit_window_seconds: int = 900  # 15 minutes

    # Lot 1.4 (§C.2.7/§D.4, AUDIT_DATALAB_2026-08-16.md) — même mécanisme
    # que le login (core/rate_limit.py::rate_limit_dependency), étendu aux
    # endpoints jusqu'ici non protégés : /register (spam de comptes),
    # uploads (tabulaire + vision), /explain (charge un modèle torch à
    # chaque appel, le plus coûteux des trois par requête).
    register_rate_limit_max_attempts: int = 10
    register_rate_limit_window_seconds: int = 3600  # 1 heure
    upload_rate_limit_max_attempts: int = 30
    upload_rate_limit_window_seconds: int = 3600  # 1 heure
    explain_rate_limit_max_attempts: int = 20
    explain_rate_limit_window_seconds: int = 3600  # 1 heure

    # Lot 1.4 (§C.2.7, AUDIT_DATALAB_2026-08-16.md) — POST /training/jobs/{id}/predict
    # acceptait un dictionnaire JSON arbitraire, sans aucune limite de
    # taille. 2 Mo est largement suffisant pour toute charge JSON légitime
    # de cette API (les payloads volumineux passent par les endpoints
    # d'upload multipart, dotés de leurs propres limites dédiées — voir
    # api/main.py::MaxJsonBodySizeMiddleware, qui ne s'applique qu'aux
    # requêtes Content-Type: application/json).
    max_json_body_size_mb: int = 2

    # Traçabilité des prédictions (Lot 5, correctif I2,
    # AUDIT_DATALAB_2026-08-16.md §I2) — au-delà de ce délai, une
    # prédiction est purgée à la prochaine requête de sa propre
    # organisation (voir services/prediction_retention.py, même principe
    # que services/job_watchdog.py : pas de scheduler dédié, purge à la
    # demande). 90 jours : assez pour investiguer un incident, sans
    # accumuler indéfiniment des données personnelles potentiellement
    # sensibles saisies dans `input_json`.
    prediction_retention_days: int = 90

    # Journalisation
    log_level: str = "INFO"

    # Observabilité (Lot 4, correctif I7, AUDIT_DATALAB_2026-08-16.md §I7).
    # Absent (None) par défaut : Sentry reste totalement inactif tant que
    # cette variable n'est pas définie — dégradation honnête, jamais un
    # crash au démarrage faute de DSN en dev/CI.
    sentry_dsn: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",  # les variables Streamlit legacy (.env historique) ne doivent pas faire planter le chargement
    )


@lru_cache
def get_settings() -> Settings:
    """Instance unique des paramètres, mise en cache pour tout le process."""
    return Settings()
