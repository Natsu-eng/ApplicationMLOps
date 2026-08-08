import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// En dev, le frontend proxy /api vers le backend FastAPI (localhost:8000) :
// pas de configuration CORS supplémentaire à gérer côté navigateur.
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
