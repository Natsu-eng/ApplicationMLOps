import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import tseslint from "typescript-eslint";

// AUDIT_ROADMAP.md, H9 — eslint était référencé dans "scripts" (package.json)
// mais jamais installé : `npm run lint` échouait immédiatement, aucune
// vérification de qualité JS/TS n'a tourné depuis des mois de lots.
export default tseslint.config(
  { ignores: ["dist"] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      // Seulement les deux règles classiques (catchent de vrais bugs :
      // hooks appelés conditionnellement, dépendances manquantes) — pas le
      // `recommended` complet du plugin v7, qui embarque des règles
      // orientées React Compiler (ex. `set-state-in-effect`) signalant
      // comme erreur le pattern standard "charger des données dans un
      // useEffect", largement et légitimement utilisé dans ce projet
      // (aucun compilateur React activé ici).
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
      // Un projet migré depuis un prototype tolère quelques `any`
      // explicites bien nommés — avertissement, pas un blocage de build.
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
    },
  }
);
