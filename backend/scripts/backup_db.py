"""Sauvegarde Postgres + stockage (datasets/modèles) — Lot 1.2 (correctif
C6, AUDIT_DATALAB_2026-08-16.md). Aucune sauvegarde n'existait avant ce lot.

Usage :
    cd backend && python -m scripts.backup_db [--output-dir backups]

Produit deux fichiers horodatés dans --output-dir :
- datalab_pg_<horodatage>.dump          (pg_dump format custom -Fc, compressé)
- datalab_storage_<horodatage>.tar.gz   (backend/storage/ complet : datasets
                                          tabulaires, datasets vision, modèles)

Nécessite `pg_dump` sur le PATH — fourni par toute installation cliente
PostgreSQL, ou exécutable directement depuis le conteneur `db` du
docker-compose (`docker compose exec db pg_dump ...`), qui embarque déjà
les outils clients de l'image officielle `postgres:15-alpine`.

Une sauvegarde jamais restaurée ne compte pas : voir `scripts/restore_db.py`
et `tests/test_backup_restore.py`, qui exerce un cycle complet
sauvegarde → restauration → vérification sur une base réelle."""
from __future__ import annotations

import argparse
import logging
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("datalab.backup")

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_database(database_url: str, output_path: Path, schema: str | None = None) -> None:
    """Dump Postgres au format custom (`-Fc`) — compressé, et seul format
    supportant la restauration sélective (table par table) via
    `pg_restore`, contrairement à un dump SQL brut (`-Fp`).

    `schema` : restreint le dump à un schéma donné (`--schema=...`) — jamais
    utilisé en usage réel (on sauvegarde toute la base), uniquement par
    `tests/test_backup_restore.py` pour exercer un cycle complet sans les
    privilèges CREATE DATABASE que le rôle applicatif n'a délibérément pas
    (moindre privilège)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["pg_dump", "-Fc", "-f", str(output_path)]
    if schema:
        cmd.append(f"--schema={schema}")
    cmd.append(database_url)
    subprocess.run(cmd, check=True)
    logger.info("[Backup] Base de données sauvegardée -> %s", output_path)


def backup_storage(storage_dir: Path, output_path: Path) -> None:
    """Archive `backend/storage/` (datasets + modèles) — un seul fichier
    plutôt qu'une copie de dossier, pour un transfert/stockage plus simple
    (S3, disque externe...)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(storage_dir, arcname="storage")
    logger.info("[Backup] Stockage sauvegardé -> %s", output_path)


def run_backup(database_url: str, storage_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    ts = _timestamp()
    db_backup_path = output_dir / f"datalab_pg_{ts}.dump"
    storage_backup_path = output_dir / f"datalab_storage_{ts}.tar.gz"
    backup_database(database_url, db_backup_path)
    backup_storage(storage_dir, storage_backup_path)
    return db_backup_path, storage_backup_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    # Import local — évite de charger toute la config applicative
    # (get_settings()) au simple import du module par les tests unitaires
    # qui appellent backup_database()/backup_storage() directement.
    from api.core.config import get_settings

    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=BACKEND_DIR / "backups")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--storage-dir", type=Path, default=BACKEND_DIR / "storage")
    args = parser.parse_args()

    db_path, storage_path = run_backup(args.database_url, args.storage_dir, args.output_dir)
    print(f"Sauvegarde terminée :\n  {db_path}\n  {storage_path}")


if __name__ == "__main__":
    main()
