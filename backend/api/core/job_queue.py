"""Files de tâches de fond — RQ + Redis.

Décision du diagnostic de migration (section D1) : CIAM n'a pas besoin de
file de tâches (ses calculs prennent 5-20 secondes, un threadpool implicite
suffit) ; un entraînement DataLab Pro avec recherche d'hyperparamètres
Optuna sur plusieurs modèles prend, lui, plusieurs dizaines de secondes à
quelques minutes — il lui faut une vraie tâche de fond, pas une requête HTTP
qui bloque.

Lot 4 (correctif I6, AUDIT_DATALAB_2026-08-16.md §I6) — une seule file
`training_queue` partagée par les 6 types de job (supervisé, clustering,
dimensionnalité, anomalies tabulaires, classification vision, anomalies
vision) faisait attendre un clustering de quelques secondes derrière un
entraînement supervisé de plusieurs minutes, même avec plusieurs workers
(`replicas: 2`, correctif C7) : rien n'empêchait deux entraînements longs
d'occuper les deux workers en même temps. Trois files, par ordre de coût
CPU/durée typique :
- `training_queue` : entraînement supervisé (recherche Optuna, le plus
  long et le plus imprévisible).
- `vision_queue` : classification et anomalies vision (torch, CPU-only,
  également long) — séparée de `training_queue` pour qu'un entraînement
  vision n'attende pas derrière un entraînement supervisé et vice-versa.
- `analysis_queue` : clustering, réduction de dimensionnalité, anomalies
  tabulaires — nettement plus courts en pratique (pas de recherche
  d'hyperparamètres), doivent rester réactifs même quand `training_queue`
  ou `vision_queue` sont occupées.

Voir `workers/run_worker.py` (quelles files un worker donné écoute,
piloté par `RQ_QUEUES`) et `docker-compose.yml` (répartition des
répliques par file).
"""
from __future__ import annotations

from redis import Redis
from rq import Queue

from api.core.config import get_settings

_settings = get_settings()

# Bug réel signalé par l'opérateur (retour direct, poste de dev sans Redis
# démarré) — `Redis.from_url()` sans timeout explicite laisse la
# CONNEXION (poignée de main TCP initiale) bloquer indéfiniment quand
# Redis est injoignable, bornée uniquement par le comportement par défaut
# de la pile TCP du système d'exploitation (souvent 20 s ou plus sous
# Windows). Concrètement observé sur `/login` : `is_rate_limited()`
# (`api/core/rate_limit.py`) est bien "échec ouvert" (Phase 1, §4) — mais
# n'échoue "ouvert" qu'APRÈS que la tentative de connexion ait fini par
# expirer, donc l'utilisateur voit "Connexion…" bloqué de longues
# secondes avant que la page ne réponde enfin.
#
# `socket_connect_timeout` seul (jamais `socket_timeout`) : borne
# UNIQUEMENT la poignée de main TCP initiale, à 3 s (largement suffisant
# pour un Redis local ou dans le même réseau Docker — jamais atteint en
# usage normal). `socket_timeout` (délai de lecture/écriture APRÈS
# connexion) n'est volontairement PAS touché ici : RQ (`Worker.work()`,
# `workers/run_worker.py`) dépend d'un `BLPOP` bloquant pour attendre un
# nouveau job sans repolling actif — lui imposer un timeout de lecture
# court casserait cette attente légitime et ferait paraître Redis "en
# panne" à chaque worker inactif entre deux jobs.
redis_conn = Redis.from_url(_settings.redis_url, socket_connect_timeout=3)

# Timeout généreux (30 min) : un entraînement à 3 modèles × recherche Optuna
# peut prendre du temps sur une machine sans GPU dédié. Vision (torch,
# CPU-only) est au moins aussi lent — même timeout.
training_queue = Queue("training", connection=redis_conn, default_timeout=1800)
vision_queue = Queue("vision", connection=redis_conn, default_timeout=1800)
# Pas de recherche d'hyperparamètres sur ces 3 types de job : un timeout
# plus court suffit et évite qu'un job bloqué monopolise longtemps le
# worker dédié à cette file.
analysis_queue = Queue("analysis", connection=redis_conn, default_timeout=600)

ALL_QUEUES = {"training": training_queue, "vision": vision_queue, "analysis": analysis_queue}
