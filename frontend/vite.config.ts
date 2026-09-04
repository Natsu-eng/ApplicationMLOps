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
    // 5300 et non le 5173 par défaut de Vite : sous Windows, Hyper-V/WSL
    // réserve dynamiquement des plages de ports, et 5141-5240 (qui contient
    // 5173) en fait partie sur ce poste — le démarrage échouait sur
    // `EACCES: permission denied 127.0.0.1:5173`, un refus du système et non
    // un port déjà occupé. Vérifiable par
    // `netsh interface ipv4 show excludedportrange protocol=tcp`.
    //
    // Déplacer le port plutôt que libérer la plage (`net stop winnat`) :
    // cette libération casse les mappages de ports Docker en cours — dont
    // Redis, dont dépendent les limites de débit, la révocation de jetons et
    // la file RQ. 5300 est franchement à l'écart des plages réservées
    // observées, contrairement à 5241 qui les jouxte.
    //
    // À garder aligné avec FRONTEND_URL côté backend (backend/.env) : c'est
    // lui qui construit les liens d'invitation et de réinitialisation.
    port: 5300,
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
