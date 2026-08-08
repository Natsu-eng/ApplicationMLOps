# workflow.md — Backend DataLab Pro : avancement lot par lot

> Mis à jour à chaque lot livré. Décisions produit issues du diagnostic de
> migration présenté et validé avant tout code : modèle **Organisation/équipe**
> pour le multi-utilisateurs, **ML classique tabulaire porté en premier**
> (avant la vision), **RQ + Redis** pour les tâches d'entraînement de fond,
> positionnement **généraliste multi-secteurs**.

## Lot 0 — Squelette (livré)

- [x] `api/main.py` : l'application FastAPI démarre, CORS configuré, `GET /api/health`
- [x] `api/core/config.py` : configuration centralisée (pydantic-settings, lecture de `backend/.env`)
- [x] `api/core/database.py` : connexion SQLAlchemy (PostgreSQL en prod, SQLite en dev), `Base` ORM prête à recevoir des modèles au Lot 1
- [x] `Dockerfile` (backend) + `docker-compose.yml` (racine) : services `db` (PostgreSQL) + `backend`
- [x] Frontend Vite + React + TypeScript + Tailwind : appelle `GET /api/health` au chargement, affiche le statut
- [ ] Aucune route métier — c'est volontaire, à construire lot par lot à partir d'ici

**Vérifié** : `uvicorn api.main:app --reload` démarre et répond sur `/api/health` ; `npm run build` du frontend passe sans erreur.

## Prochains lots (résumé — détail complet dans le diagnostic de migration)

| Lot | Contenu | Livrable testable |
|---|---|---|
| 1 | Authentification (JWT + bcrypt, pattern CIAM) + modèle Organisation/équipe | Inscription/connexion, route protégée, isolation par organisation |
| 2 | Upload + gestion de datasets tabulaires | Un CSV uploadé apparaît dans "Mes données" |
| 3 | Entraînement ML classique de bout en bout (RQ + Redis, suivi de progression) | Un modèle sklearn s'entraîne en tâche de fond, progression visible |
| 4-5 | Évaluation/visualisation ML classique, catalogue complet | Parité fonctionnelle avec l'app Streamlit historique |
| 6-8 | Upload / entraînement / évaluation vision (détection d'anomalies) | Parité fonctionnelle côté vision |
| 9 | Registre de modèles unifié (versioning) | Remplace les 3 mécanismes de persistance de l'app historique |
| 10 | Durcissement SaaS (erreurs, audit, quotas) | Prêt pour un client pilote |

Ce fichier sera complété à chaque lot livré avec le détail réel (fichiers
créés, endpoints exposés, décisions techniques prises en cours de route) —
même format que le `workflow.md` de CIAM, sourcé fichier par fichier.
