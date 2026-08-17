import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// En dev, le frontend proxy /api vers le backend FastAPI (localhost:8000) :
// pas de configuration CORS supplémentaire à gérer côté navigateur.
//
// Une SEULE entrée /api (correctif — incident réel de déploiement) : tous
// les endpoints backend sont désormais préfixés /api (api/main.py), donc
// une seule entrée de proxy couvre tout le client. Avant ce correctif, des
// entrées séparées par domaine métier (/training, /clustering, /datasets,
// /vision, ...) interceptaient aussi les ROUTES DE PAGE de même nom
// (`/training`, `/clustering`, `/datasets`, `/vision/classification`...) :
// rafraîchir ou ouvrir un lien direct sur ces pages en dev renvoyait le
// JSON d'erreur du backend au lieu du HTML de la SPA, puisque Vite
// proxifiait la requête de navigation elle-même avant que React Router ne
// la voie jamais.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // IPv4 explicite — évite un bind ::1 uniquement qui rendrait le serveur
    // injoignable via 127.0.0.1 selon la résolution DNS locale de "localhost".
    host: "127.0.0.1",
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
