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
    // Port par défaut de Vite — à garder aligné avec FRONTEND_URL
    // (backend/.env), qui construit les liens d'invitation et de
    // réinitialisation de mot de passe : un port désaccordé donnerait des
    // liens morts sans aucune erreur visible.
    //
    // Sous Windows, si le démarrage échoue sur `EACCES: permission denied
    // 127.0.0.1:5173`, ce n'est PAS un port occupé (ce serait EADDRINUSE)
    // mais une plage réservée dynamiquement par Hyper-V/WSL — 5141-5240
    // contient 5173. Diagnostic :
    //   netsh interface ipv4 show excludedportrange protocol=tcp
    // Remède (PowerShell administrateur), qui libère le port et le réserve
    // pour nous avant que Hyper-V ne le reprenne au prochain démarrage :
    //   net stop winnat
    //   netsh int ipv4 add excludedportrange protocol=tcp startport=5173 numberofports=1
    //   net start winnat
    // `winnat` coupe les mappages de ports Docker au passage : redémarrer
    // les conteneurs ensuite (`docker restart datalab_redis`).
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
