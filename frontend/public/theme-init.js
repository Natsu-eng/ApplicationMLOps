// Pose data-theme AVANT le premier rendu React — évite un flash de thème
// incorrect au chargement (voir ThemeContext.tsx::resolveInitialTheme, qui
// DOIT rester en miroir exact de cette résolution : localStorage →
// prefers-color-scheme → graphite). La préférence serveur ne peut pas être
// lue ici de façon synchrone (pas de SSR) — ThemeProvider la récupère juste
// après montage et bascule sans rechargement si besoin.
//
// Fichier séparé plutôt qu'un <script> inline dans index.html (Phase 1,
// AUDIT_BACKEND_2026-08-23.md, Axe E) — la CSP stricte ajoutée en Phase 1
// (`script-src 'self'`, sans `unsafe-inline`) bloquerait un script inline ;
// un fichier statique servi par la même origine passe `script-src 'self'`
// sans affaiblir la CSP.
(function () {
  var VALID = ["graphite", "ivoire", "minuit", "ardoise", "porcelaine"];
  try {
    var stored = localStorage.getItem("datalab_theme");
    var theme = VALID.indexOf(stored) !== -1 ? stored
      : (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches === false)
        ? "ivoire"
        : "graphite";
    document.documentElement.setAttribute("data-theme", theme);
  } catch (e) {
    document.documentElement.setAttribute("data-theme", "graphite");
  }
})();
