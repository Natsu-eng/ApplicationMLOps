"""File de tâches pour l'entraînement — RQ + Redis.

Décision du diagnostic de migration (section D1) : CIAM n'a pas besoin de
file de tâches (ses calculs prennent 5-20 secondes, un threadpool implicite
suffit) ; un entraînement DataLab Pro avec recherche d'hyperparamètres
Optuna sur plusieurs modèles prend, lui, plusieurs dizaines de secondes à
quelques minutes — il lui faut une vraie tâche de fond, pas une requête HTTP
qui bloque.
"""
from __future__ import annotations

from redis import Redis
from rq import Queue

from api.core.config import get_settings

_settings = get_settings()

redis_conn = Redis.from_url(_settings.redis_url)

# Timeout généreux (30 min) : un entraînement à 3 modèles × recherche Optuna
# peut prendre du temps sur une machine sans GPU dédié.
training_queue = Queue("training", connection=redis_conn, default_timeout=1800)
