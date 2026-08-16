"""Restauration Postgres + stockage depuis une sauvegarde produite par
`scripts/backup_db.py` — Lot 1.2 (correctif C6, AUDIT_DATALAB_2026-08-16.md).

Usage :
    cd backend && python -m scripts.restore_db \\
        --db-dump backups/datalab_pg_20260816T220000Z.dump \\
        --storage-archive backups/datalab_storage_20260816T220000Z.tar.gz \\
        --database-url postgresql://... \\
        --confirm

Sécurité délibérée : `--confirm` est OBLIGATOIRE pour exécuter la
restauration (sinon le script ne fait qu'afficher ce qu'il ferait) — une
restauration écrase la base cible (`pg_restore --clean`), un geste
destructif qui ne doit jamais être déclenché par accident (ex. mauvais
copier-coller de commande)."""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

logger = logging.getLogger("datalab.restore")

BACKEND_DIR = Path(__file__).resolve().parent.parent


def restore_database(dump_path: Path, database_url: str) -> None:
    """`--clean --if-exists` : supprime les objets existants avant de les
    recréer depuis le dump — une restauration doit remplacer l'état de la
    base cible, pas fusionner avec son contenu actuel (qui pourrait être
    corrompu ou incomplet, raison même de la restauration)."""
    subprocess.run(
        ["pg_restore", "--clean", "--if-exists", "--no-owner", "-d", database_url, str(dump_path)],
        check=True,
    )
    logger.info("[Restore] Base de données restaurée depuis %s", dump_path)


def restore_storage(archive_path: Path, target_dir: Path) -> None:
    """Extrait l'archive DANS `target_dir`, quel que soit son nom — l'archive
    contient toujours `storage/` comme racine interne fixe (voir
    `backup_db.py::backup_storage`), donc une extraction directe créerait un
    dossier `storage/` au lieu de peupler `target_dir` lui-même. Remplace le
    contenu existant à chaque nom d'entrée en conflit (restauration =
    remplacement, pas fusion — même sémantique que `restore_database`)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        extracted_root = tmp / "storage"
        for item in extracted_root.iterdir():
            dest = target_dir / item.name
            if dest.is_dir():
                shutil.rmtree(dest)
            elif dest.exists():
                dest.unlink()
            shutil.move(str(item), str(dest))
    logger.info("[Restore] Stockage restauré -> %s", target_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    from api.core.config import get_settings

    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-dump", type=Path, required=True)
    parser.add_argument("--storage-archive", type=Path, default=None, help="Optionnel — restauration DB seule si omis")
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--storage-dir", type=Path, default=BACKEND_DIR / "storage")
    parser.add_argument("--confirm", action="store_true", help="Exécute réellement la restauration (sinon, aperçu seul)")
    args = parser.parse_args()

    print(f"Restauration prévue :\n  DB dump      : {args.db_dump}\n  Cible DB     : {args.database_url}")
    if args.storage_archive:
        print(f"  Archive stockage : {args.storage_archive}\n  Cible stockage   : {args.storage_dir}")

    if not args.confirm:
        print("\n--confirm absent : aucune action effectuée (aperçu seul).")
        return

    restore_database(args.db_dump, args.database_url)
    if args.storage_archive:
        restore_storage(args.storage_archive, args.storage_dir)
    print("Restauration terminée.")


if __name__ == "__main__":
    main()
