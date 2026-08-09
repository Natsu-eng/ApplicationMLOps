# DataLab Pro — Frontend (React + TypeScript)

SPA du SaaS DataLab Pro. Stack et conventions reprises de CIAM
(`concrete-ai-platform/frontend`), déjà éprouvée en production.

## État d'avancement

> **Lot 3 (entraînement ML supervisé) — état actuel.** Connexion, inscription,
> tableau de bord avec équipe, "Mes données", et page "Entraînement" :
> configuration d'un job, suivi de progression en direct (polling), résultat
> détaillé (métriques, importance SHAP, intervalles conformes CQR) — voir
> [`../backend/workflow.md`](../backend/workflow.md).

## Système de design

Composants partagés dans `src/components/ui/` (`Card`, `Button`, `Badge`,
`Avatar`, `Input`, `Modal`) — même style dans toutes les pages : fond dégradé
teal sur quasi-noir (`index.css`), cartes en verre dépoli (`backdrop-blur`),
accent teal, badges de statut sémantiques (succès/avertissement/danger).
Icônes : `lucide-react`. Objectif explicite : un niveau de finition visuelle
au-dessus de CIAM, pas seulement fonctionnel.

## Stack

| Techno | Version |
| --- | --- |
| React / React-DOM | 19.1.0 |
| TypeScript | 5.8.3 |
| Vite | 6.3.5 |
| Tailwind CSS | 4.1.7 |

Pas d'axios (fetch natif via `src/api/client.ts`). Pas de librairie de state
management globale pour l'instant — état local + React Context au fur et à
mesure des besoins (auth au Lot 1).

## Démarrage local

```bash
cd frontend
npm install
cp .env.example .env.local   # laisser VITE_API_URL vide en dev
npm run dev
# → http://localhost:5173 (le backend doit tourner sur le port 8000)
```

## Structure

```text
frontend/
├── src/
│   ├── main.tsx                 ← point d'entrée
│   ├── App.tsx                    ← routes (react-router-dom)
│   ├── api/client.ts                ← client API centralisé (fetch natif, JWT Bearer, upload multipart)
│   ├── contexts/AuthContext.tsx       ← session (token localStorage, /auth/me au chargement)
│   ├── components/
│   │   ├── AppShell.tsx                 ← en-tête + navigation commune aux pages protégées
│   │   ├── ProtectedRoute.tsx             ← garde de route (redirige vers /login)
│   │   └── ui/                              ← système de design (Card, Button, Badge, Avatar, Input, Modal)
│   ├── pages/
│   │   ├── Login.tsx                          ← connexion
│   │   ├── Register.tsx                         ← inscription (crée une organisation)
│   │   ├── Dashboard.tsx                          ← profil + équipe + ajout de membre (owner)
│   │   ├── Datasets.tsx                             ← upload, liste, aperçu des datasets
│   │   └── Training.tsx                               ← configuration + suivi + historique d'entraînements
│   ├── components/training/ModelResultModal.tsx        ← métriques, barres SHAP, couverture CQR, fiche modèle
│   ├── utils/format.ts                                ← formatage taille/date/métriques
│   └── index.css                                        ← Tailwind + fond dégradé global
├── index.html
├── package.json
├── vite.config.ts               ← proxy /api, /auth, /datasets, /training → backend en dev
├── tsconfig*.json
├── .env.example
└── README.md                       ← ce fichier
```

## Build production

```bash
npm run build   # → dist/
```
