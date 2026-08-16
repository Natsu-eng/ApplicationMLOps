# Sauvegarde / restauration (Lot 1.2, correctif C6)

Deux objets à sauvegarder séparément : la base Postgres (métadonnées, jobs,
métriques) et `backend/storage/` (fichiers de datasets et modèles entraînés,
non versionnés en base).

## Sauvegarder

```bash
cd backend
python -m scripts.backup_db --output-dir backups
```

Produit `backups/datalab_pg_<horodatage>.dump` et
`backups/datalab_storage_<horodatage>.tar.gz`. Nécessite `pg_dump` sur le
PATH — fourni par toute installation cliente PostgreSQL, ou exécutable
depuis le conteneur `db` du `docker-compose.yml` :

```bash
docker compose exec db pg_dump -Fc "$DATABASE_URL" > backup.dump
```

À planifier (cron/tâche planifiée) selon la fréquence acceptable de perte
de données pour votre déploiement — aucune fréquence n'est imposée par ce
script, c'est une décision opérationnelle propre à chaque installation.

## Restaurer

```bash
cd backend
python -m scripts.restore_db \
    --db-dump backups/datalab_pg_20260816T220000Z.dump \
    --storage-archive backups/datalab_storage_20260816T220000Z.tar.gz \
    --database-url postgresql://... \
    --confirm
```

`--confirm` est obligatoire : sans lui, le script affiche seulement ce
qu'il ferait (aperçu). La restauration DB utilise `pg_restore --clean
--if-exists` — elle remplace le contenu de la base cible, jamais une
fusion avec son état actuel.

**Ne restaurez jamais directement sur une base en production sans un
second avis** — dans le doute, restaurez d'abord dans une base de
vérification, contrôlez les données, puis basculez.

## Preuve que ça fonctionne

`tests/test_backup_restore.py` exerce un cycle complet (peuple → sauvegarde
→ supprime → restaure → vérifie) sur un vrai Postgres, dans un schéma
jetable dédié (jamais les données réelles). Un script de sauvegarde jamais
restauré ne compte pas — voir DECISIONS.md, section Lot 1.
