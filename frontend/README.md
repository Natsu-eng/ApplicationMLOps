# DataLab Pro — Frontend (React + TypeScript)

SPA du SaaS DataLab Pro. Stack et conventions reprises de CIAM
(`concrete-ai-platform/frontend`), déjà éprouvée en production.

## État d'avancement

> **Lot 0 (squelette) — état actuel.** Une seule page : vérifie la connexion
> au backend (`GET /api/health`) et affiche le statut. Aucune vraie page
> métier (connexion, upload, entraînement...) n'existe encore — voir
> [`../backend/workflow.md`](../backend/workflow.md) pour l'avancement.

## Stack

| Techno | Version |
|---|---|
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

```
frontend/
├── src/
│   ├── main.tsx        ← point d'entrée
│   ├── App.tsx           ← page unique du Lot 0 (statut backend)
│   ├── api/client.ts       ← client API centralisé (fetch natif)
│   └── index.css             ← import Tailwind
├── index.html
├── package.json
├── vite.config.ts               ← proxy /api → backend en dev
├── tsconfig*.json
├── .env.example
└── README.md                       ← ce fichier
```

## Build production

```bash
npm run build   # → dist/
```
