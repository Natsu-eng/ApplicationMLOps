import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

export type ThemePreference = "light" | "dark" | "system";

const STORAGE_KEY = "datalab_theme";

interface ThemeContextValue {
  /** Choix explicite de l'utilisateur — "system" = suit `prefers-color-scheme`, jamais forcé. */
  preference: ThemePreference;
  setPreference: (pref: ThemePreference) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function readStoredPreference(): ThemePreference {
  const stored = localStorage.getItem(STORAGE_KEY);
  return stored === "light" || stored === "dark" ? stored : "system";
}

/** Applique `data-theme` sur `<html>` — "system" retire l'attribut (index.css
 * suit alors `prefers-color-scheme` seul), "light"/"dark" le posent
 * explicitement pour gagner sur la préférence système dans les deux sens. */
function applyThemeAttribute(pref: ThemePreference) {
  if (pref === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.setAttribute("data-theme", pref);
  }
}

/** Préférence de thème (clair/sombre/système), persistée (`localStorage`,
 * pas de compte à travers les appareils pour l'instant) — retour utilisateur
 * direct ("ajoute un mode sombre"). Appliquée en dehors du cycle de rendu
 * React (attribut DOM direct) pour éviter tout flash de thème incorrect au
 * premier rendu. */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [preference, setPreferenceState] = useState<ThemePreference>(readStoredPreference);

  useEffect(() => {
    applyThemeAttribute(preference);
  }, [preference]);

  const setPreference = useCallback((pref: ThemePreference) => {
    setPreferenceState(pref);
    localStorage.setItem(STORAGE_KEY, pref);
  }, []);

  return <ThemeContext.Provider value={{ preference, setPreference }}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme() doit être appelé à l'intérieur de <ThemeProvider>");
  return ctx;
}
