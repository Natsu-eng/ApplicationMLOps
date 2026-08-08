# DataLab Pro — Frontend (React + TypeScript)

SPA du SaaS DataLab Pro. Stack et conventions reprises de CIAM
(`concrete-ai-platform/frontend`), déjà éprouvée en production.

## État d'avancement

> **Lot 1 (authentification + organisations) — état actuel.** Connexion,
> inscription (création d'organisation), tableau de bord protégé avec
> gestion d'équipe. Upload de données et entraînement arrivent aux lots
> suivants — voir [`../backend/workflow.md`](../backend/workflow.md).

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
│   ├── api/client.ts                ← client API centralisé (fetch natif, JWT Bearer)
│   ├── contexts/AuthContext.tsx       ← session (token localStorage, /auth/me au chargement)
│   ├── components/ProtectedRoute.tsx    ← garde de route (redirige vers /login)
│   ├── pages/
│   │   ├── Login.tsx                      ← connexion
│   │   ├── Register.tsx                     ← inscription (crée une organisation)
│   │   └── Dashboard.tsx                      ← profil + équipe + ajout de membre (owner)
│   └── index.css                                ← import Tailwind
├── index.html
├── package.json
├── vite.config.ts               ← proxy /api, /auth → backend en dev
├── tsconfig*.json
├── .env.example
└── README.md                       ← ce fichier
```

## Build production

```bash
npm run build   # → dist/
```
