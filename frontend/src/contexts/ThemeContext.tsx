import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, getToken, type UiTheme } from "../api/client";

export type { UiTheme };

const STORAGE_KEY = "datalab_theme";
const VALID_THEMES: UiTheme[] = ["graphite", "ivoire", "minuit", "ardoise", "porcelaine"];
const DEFAULT_THEME: UiTheme = "graphite";

function isUiTheme(value: string | null): value is UiTheme {
  return value !== null && (VALID_THEMES as string[]).includes(value);
}

/** Résolution du thème initial (avant tout appel réseau) — même ordre que le
 * script inline d'index.html, dans lequel cette fonction doit rester en
 * miroir exact (voir ce fichier) : localStorage → préférence système
 * (sombre → graphite, clair → ivoire) → graphite. La préférence serveur
 * n'entre en jeu qu'après montage (voir `ThemeProvider`), une fois le
 * profil chargé — impossible à lire de façon synchrone avant le premier
 * rendu sans SSR. */
function resolveInitialTheme(): UiTheme {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (isUiTheme(stored)) return stored;
  if (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches === false) {
    return "ivoire";
  }
  return DEFAULT_THEME;
}

function applyThemeAttribute(theme: UiTheme) {
  document.documentElement.setAttribute("data-theme", theme);
}

interface ThemeContextValue {
  theme: UiTheme;
  /** Change le thème immédiatement (DOM + localStorage), et le persiste sur
   * le profil serveur si un utilisateur est connecté — sans jamais recharger
   * la page. */
  setTheme: (theme: UiTheme) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

/** Thème d'interface (5 directions, voir _design/SPEC-UI.md §2) — ordre de
 * résolution demandé : préférence enregistrée côté serveur (profil
 * utilisateur) → localStorage → prefers-color-scheme → graphite. Le serveur
 * gagne dès qu'il répond (l'utilisateur peut changer de poste), mais
 * n'empêche jamais un rendu immédiat non plus : on démarre avec la
 * résolution locale posée par le script inline d'index.html, sans flash. */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<UiTheme>(() => {
    const attr = document.documentElement.getAttribute("data-theme");
    return isUiTheme(attr) ? attr : resolveInitialTheme();
  });

  useEffect(() => {
    applyThemeAttribute(theme);
  }, [theme]);

  // Préférence serveur — ne s'applique qu'une fois, au montage, si un token
  // existe déjà. Échec silencieux (token invalide, hors ligne...) : le thème
  // résolu localement reste en place, jamais d'écran sans couleurs.
  useEffect(() => {
    if (!getToken()) return;
    let cancelled = false;
    api.users
      .preferences()
      .then((prefs) => {
        if (!cancelled && isUiTheme(prefs.ui_theme)) {
          setThemeState(prefs.ui_theme);
          localStorage.setItem(STORAGE_KEY, prefs.ui_theme);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const setTheme = useCallback((next: UiTheme) => {
    setThemeState(next);
    localStorage.setItem(STORAGE_KEY, next);
    if (getToken()) {
      api.users.updatePreferences({ ui_theme: next }).catch(() => {});
    }
  }, []);

  return <ThemeContext.Provider value={{ theme, setTheme }}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme() doit être appelé à l'intérieur de <ThemeProvider>");
  return ctx;
}
