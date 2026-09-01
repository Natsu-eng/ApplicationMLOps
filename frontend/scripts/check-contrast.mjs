// Vérification programmatique de contraste (Lot 11 — Vérification finale).
// Contrairement aux scripts axe-core (Playwright, nécessitent un serveur
// vivant), ce script ne dépend que des fichiers source : il tourne dans la
// CI sans navigateur ni backend, et échoue (exit 1) si une combinaison
// texte/fond réellement utilisée par l'app retombe sous 4,5:1 dans l'un des
// 5 thèmes — exactement la classe de bug trouvée et corrigée au Lot 11
// (ColorIconBadge.tsx, Badge.tsx : opacités de fond /4 insuffisantes une
// fois composées avec un fond déjà teinté).
//
// Les valeurs sont EXTRAITES des vrais fichiers source (regex), jamais
// dupliquées à la main : si quelqu'un change une couleur ou une opacité
// sans repasser par ce script, la CI le détecte.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(__dirname, "..", "src");

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}
function relLum([r, g, b]) {
  const f = (c) => {
    const cs = c / 255;
    return cs <= 0.03928 ? cs / 12.92 : ((cs + 0.055) / 1.055) ** 2.4;
  };
  const [rl, gl, bl] = [f(r), f(g), f(b)];
  return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl;
}
function contrast(hexA, hexB) {
  const la = relLum(hexToRgb(hexA)) + 0.05;
  const lb = relLum(hexToRgb(hexB)) + 0.05;
  return la > lb ? la / lb : lb / la;
}
function alphaBlend(fgHex, bgHex, alpha) {
  const fg = hexToRgb(fgHex);
  const bg = hexToRgb(bgHex);
  const mixed = fg.map((c, i) => Math.round(c * alpha + bg[i] * (1 - alpha)));
  return "#" + mixed.map((c) => c.toString(16).padStart(2, "0")).join("");
}

// --- 1. Extraire les hex par thème depuis themes.css (commentaires /* #xxxxxx */
//        pour les tokens OKLCH, valeurs hex littérales pour --s1…--s6) ---
const themesCss = readFileSync(path.join(SRC, "styles", "themes.css"), "utf-8");
const THEME_NAMES = ["graphite", "ivoire", "minuit", "ardoise", "porcelaine"];
const TOKENS = ["canvas", "surface", "raised", "text-muted", "warning", "danger"];

function extractThemeBlock(css, theme) {
  const start = css.indexOf(`[data-theme="${theme}"]`);
  const end = css.indexOf("\n}", start);
  return css.slice(start, end);
}
function extractHexComment(block, varName) {
  const re = new RegExp(`--${varName}:\\s*oklch\\([^)]*\\)\\s*;\\s*/\\*\\s*(#[0-9a-fA-F]{6})`);
  const m = block.match(re);
  if (!m) throw new Error(`Jeton --${varName} introuvable (commentaire hex manquant) dans themes.css`);
  return m[1];
}
function extractSeriesHex(block, n) {
  const re = new RegExp(`--s${n}:\\s*(#[0-9a-fA-F]{6})`);
  const m = block.match(re);
  if (!m) throw new Error(`Jeton --s${n} introuvable dans themes.css`);
  return m[1];
}

const themes = {};
for (const theme of THEME_NAMES) {
  const block = extractThemeBlock(themesCss, theme);
  themes[theme] = {
    canvas: extractHexComment(block, "canvas"),
    raised: extractHexComment(block, "raised"),
    surface: extractHexComment(block, "surface"),
    textMuted: extractHexComment(block, "text-muted"),
    warning: extractHexComment(block, "warning"),
    danger: extractHexComment(block, "danger"),
    s1: extractSeriesHex(block, 1),
    s2: extractSeriesHex(block, 2),
    s3: extractSeriesHex(block, 3),
    s4: extractSeriesHex(block, 4),
    s1Ink: extractInk(block, 1),
    s2Ink: extractInk(block, 2),
    s3Ink: extractInk(block, 3),
    s4Ink: extractInk(block, 4),
  };
}

// --- 2. Extraire les opacités réellement utilisées (ColorIconBadge.tsx,
//        Badge.tsx) — jamais des pourcentages recopiés à la main ---
/** Encre de série : --sN-ink est déclarée en oklch() suivie de son hex en
 *  commentaire, comme tous les autres jetons — on réutilise donc
 *  `extractHexComment`, sans convertir oklch → sRGB ici. */
function extractInk(block, n) {
  try {
    return extractHexComment(block, `s${n}-ink`);
  } catch {
    return null;
  }
}

function extractOpacity(fileContent, bgToken) {
  // ex. "bg-accent-1/3" -> 3, "bg-warning/4" -> 4
  const re = new RegExp(`bg-${bgToken}/(\\d+(?:\\.\\d+)?)`);
  const m = fileContent.match(re);
  if (!m) throw new Error(`Classe bg-${bgToken}/N introuvable`);
  return Number(m[1]) / 100;
}

const colorIconBadge = readFileSync(path.join(SRC, "components", "ui", "ColorIconBadge.tsx"), "utf-8");
const badge = readFileSync(path.join(SRC, "components", "ui", "Badge.tsx"), "utf-8");

const accentSurfaceOpacity = {
  s1: extractOpacity(colorIconBadge, "accent-1"),
  s2: extractOpacity(colorIconBadge, "accent-2"),
  s3: extractOpacity(colorIconBadge, "accent-3"),
  s4: extractOpacity(colorIconBadge, "accent-4"),
  danger: extractOpacity(colorIconBadge, "destructive"),
};
const badgeOpacity = {
  warning: extractOpacity(badge, "warning"),
  danger: extractOpacity(badge, "destructive"),
};

// --- 3. Vérifier : text-muted-foreground sur une carte teintée
//        (accentSurfaceClass) + text-warning/text-destructive sur un Badge,
//        contre la surface nue de CHAQUE thème (cas de base, sans double
//        lavis d'une ligne de tableau déjà teintée — ce cas plus rare et
//        plus marginal reste documenté à la main dans RAPPORT-FINAL.md) ---
const MIN_RATIO = 4.5;
let failures = [];

for (const [theme, t] of Object.entries(themes)) {
  for (const key of ["s1", "s2", "s3", "s4", "danger"]) {
    const accentHex = key === "danger" ? t.danger : t[key];
    const tinted = alphaBlend(accentHex, t.surface, accentSurfaceOpacity[key]);
    const ratio = contrast(t.textMuted, tinted);
    if (ratio < MIN_RATIO) {
      failures.push(`${theme}: text-muted-foreground sur accentSurfaceClass(${key}) = ${ratio.toFixed(2)}:1 (minimum ${MIN_RATIO}:1)`);
    }
  }
  for (const [variant, hex] of [["warning", t.warning], ["danger", t.danger]]) {
    const tinted = alphaBlend(hex, t.surface, badgeOpacity[variant]);
    const ratio = contrast(hex, tinted);
    if (ratio < MIN_RATIO) {
      failures.push(`${theme}: Badge variant="${variant}" (texte sur son propre fond) = ${ratio.toFixed(2)}:1 (minimum ${MIN_RATIO}:1)`);
    }
  }
}

// --- 4. Vérifier les ENCRES de série (--sN-ink) employées comme TEXTE ---
//        Contrôle qui manquait. La revue finale a trouvé la tuile
//        « Analyses ML » du tableau de bord affichant le compte Vision
//        (pilier teal → --s3) en `text-h2` (18px/600 — donc PAS du « texte
//        large » au sens WCAG, qui exige ≥24px ou ≥18,66px ET gras 700)
//        à 4,18-4,27:1 en ivoire et porcelaine. Ni ce script ni le scan axe
//        ne le voyaient : ce script ne testait pas les séries en texte, et
//        le scan axe ne lisait que `violations`, jamais `incomplete`.
const valueBlock = colorIconBadge.slice(colorIconBadge.indexOf("ACCENT_VALUE_TEXT_CLASSES"));
const rawSeriesAsText = valueBlock.match(/text-accent-[1-4](?!-ink)/g);
if (rawSeriesAsText) {
  failures.push(
    `ColorIconBadge: ${rawSeriesAsText.join(", ")} employé comme couleur de texte — ` +
      "les couleurs de série sont calibrées pour des marques de graphique (3:1), " +
      "pas pour du texte (4,5:1). Utiliser text-accent-N-ink.",
  );
}

for (const [theme, t] of Object.entries(themes)) {
  for (const bgName of ["canvas", "surface", "raised"]) {
    for (let i = 1; i <= 4; i++) {
      const ink = t[`s${i}Ink`];
      if (!ink) {
        failures.push(`${theme}: jeton --s${i}-ink absent de themes.css`);
        continue;
      }
      const ratio = contrast(ink, t[bgName]);
      if (ratio < MIN_RATIO) {
        failures.push(`${theme}: --s${i}-ink sur ${bgName} = ${ratio.toFixed(2)}:1 (minimum ${MIN_RATIO}:1)`);
      }
    }
  }
}

if (failures.length > 0) {
  console.error(`✖ ${failures.length} combinaison(s) texte/fond sous ${MIN_RATIO}:1 :\n`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}

console.log(
  `✓ Contraste programmatique : text-muted-foreground/accentSurfaceClass, Badge (warning/danger) ` +
    `et les 4 encres de série sur canvas/surface/raised atteignent toutes ≥ ${MIN_RATIO}:1 ` +
    `sur les ${THEME_NAMES.length} thèmes ; aucune couleur de série employée comme texte.`,
);
