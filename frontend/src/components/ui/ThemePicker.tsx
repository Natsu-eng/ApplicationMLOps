import { Check } from "lucide-react";
import { useTheme } from "../../contexts/ThemeContext";
import type { UiTheme } from "../../api/client";

/** Métadonnées des 5 thèmes (SPEC-UI.md §2) — nom et phrase de positionnement
 * copiés tels quels depuis les commentaires d'en-tête de chaque bloc dans
 * `_design/themes.css` (source de vérité, ne pas reformuler ici). Le
 * contraste minimum est celui que `_design/tune.py` a mesuré pour ce thème
 * (même document) — recalculé mécaniquement à chaque régénération de
 * themes.css, jamais estimé. */
export const THEME_META: Record<UiTheme, { name: string; tagline: string; minContrast: string }> = {
  graphite: {
    name: "Graphite & Ambre",
    tagline:
      "Chaleur minérale. Ambre sur graphite — la direction qui vous distingue le plus : vos concurrents sont tous bleus.",
    minContrast: "5,31:1",
  },
  ivoire: {
    name: "Ivoire & Encre",
    tagline: "Clair, sobre, imprimable. Vert-sarcelle profond sur ivoire — pour les captures d'écran en rapport et les salles éclairées.",
    minContrast: "4,52:1",
  },
  minuit: {
    name: "Minuit & Iris",
    tagline: "Nuit froide, accent iris. La plus familière aux profils techniques sans tomber dans le bleu générique.",
    minContrast: "5,25:1",
  },
  ardoise: {
    name: "Ardoise & Chaux",
    tagline: "Ardoise et vert chaux. La plus contrastée des cinq — un accent qu'on ne rate pas, y compris sur vidéoprojecteur.",
    minContrast: "5,35:1",
  },
  porcelaine: {
    name: "Porcelaine & Cobalt",
    tagline: "Porcelaine et cobalt. La plus institutionnelle — celle qui passe sans discussion dans un grand compte.",
    minContrast: "4,55:1",
  },
};

export const THEME_ORDER: UiTheme[] = ["graphite", "ivoire", "minuit", "ardoise", "porcelaine"];

/** Aperçu réel d'un thème (fond/carte/accent/texte) — rendu en posant
 * `data-theme` sur un conteneur isolé, jamais des couleurs recopiées à la
 * main : si themes.css change, l'aperçu suit automatiquement. */
function ThemeSwatch({ theme }: { theme: UiTheme }) {
  return (
    <div data-theme={theme} className="grid grid-cols-4 gap-1 rounded-md overflow-hidden" aria-hidden="true">
      <div className="h-8" style={{ background: "var(--canvas)" }} />
      <div className="h-8" style={{ background: "var(--surface)" }} />
      <div className="h-8" style={{ background: "var(--accent)" }} />
      <div className="h-8 flex items-center justify-center" style={{ background: "var(--surface)" }}>
        <span className="text-[10px] font-semibold" style={{ color: "var(--text)" }}>
          Aa
        </span>
      </div>
    </div>
  );
}

/** Sélecteur complet (5 vignettes) — page Préférences. Chaque vignette est
 * un vrai bouton radio (un seul choix actif), navigable au clavier comme
 * n'importe quel groupe de boutons natifs. */
export function ThemePickerGrid() {
  const { theme, setTheme } = useTheme();

  return (
    <div role="radiogroup" aria-label="Thème d'interface" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {THEME_ORDER.map((id) => {
        const meta = THEME_META[id];
        const active = theme === id;
        return (
          <button
            key={id}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => setTheme(id)}
            className={`text-left rounded-2xl border p-3 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-[var(--accent)] ${
              active ? "border-primary" : "border-border hover:bg-muted/50"
            }`}
            style={active ? { borderColor: "var(--accent)" } : undefined}
          >
            <ThemeSwatch theme={id} />
            <div className="mt-2.5 flex items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="text-h3 text-foreground flex items-center gap-1.5">
                  {meta.name}
                  {active && <Check size={14} className="text-primary flex-shrink-0" />}
                </p>
                <p className="text-caption text-muted-foreground mt-1 leading-relaxed">{meta.tagline}</p>
              </div>
            </div>
            <p className="ov mt-2">{meta.minContrast} min</p>
          </button>
        );
      })}
    </div>
  );
}

/** Variante compacte — menu de l'avatar : une ligne par thème, coche sur le
 * thème actif, pas de phrase de positionnement (place limitée). Destinée à
 * être posée à l'intérieur d'un conteneur `role="menu"` (voir AppShell) :
 * `role="group"` ici, jamais `radiogroup` — un `menuitemradio` exige un
 * parent `menu`/`menubar`/`group`, pas `radiogroup` (règle axe-core
 * aria-required-parent, détectée par _design/JOURNAL.md Lot 1). */
export function ThemePickerCompact() {
  const { theme, setTheme } = useTheme();

  return (
    <div role="group" aria-label="Thème d'interface" className="flex flex-col gap-1">
      {THEME_ORDER.map((id) => {
        const meta = THEME_META[id];
        const active = theme === id;
        return (
          <button
            key={id}
            type="button"
            role="menuitemradio"
            aria-checked={active}
            onClick={() => setTheme(id)}
            className="flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-foreground hover:bg-muted transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[var(--accent)]"
          >
            <span
              data-theme={id}
              className="h-4 w-4 flex-shrink-0 rounded-full border"
              style={{ background: "var(--accent)", borderColor: "var(--border-strong)" }}
              aria-hidden="true"
            />
            <span className="flex-1 truncate text-left">{meta.name}</span>
            {active && <Check size={14} className="text-primary flex-shrink-0" />}
          </button>
        );
      })}
    </div>
  );
}
