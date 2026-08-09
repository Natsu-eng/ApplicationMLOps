# DataLab Pro — Frontend (React + TypeScript)

SPA du SaaS DataLab Pro. Stack et conventions reprises de CIAM
(`concrete-ai-platform/frontend`), déjà éprouvée en production.

## État d'avancement

> **Lot 2 (datasets tabulaires) — état actuel.** Connexion, inscription
> (création d'organisation), tableau de bord avec gestion d'équipe, page
> "Mes données" (upload par glisser-déposer, grille de cartes, aperçu).
> Entraînement arrive au Lot 3 — voir
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
│   │   └── Datasets.tsx                             ← upload, liste, aperçu des datasets
│   ├── utils/format.ts                                ← formatage taille de fichier / date
│   └── index.css                                        ← Tailwind + fond dégradé global
├── index.html
├── package.json
├── vite.config.ts               ← proxy /api, /auth, /datasets → backend en dev
├── tsconfig*.json
├── .env.example
└── README.md                       ← ce fichier
```

## Build production

```bash
npm run build   # → dist/
```
