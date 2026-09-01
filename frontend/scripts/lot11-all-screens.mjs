import { chromium } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

// Lot 11 (Vérification finale) — capture des 20 écrans × 5 thèmes +
// axe-core sur chacun. Les 2 pages publiques (login/register) n'ont pas de
// jeton ; les 18 autres utilisent le compte réel org #2 (données réelles,
// pas de fixtures). Les vues de RÉSULTAT des 5 domaines non
// supervisés/vision + Entraînement ont déjà été capturées séparément
// (scripts/lot10-results-themes-axe.mjs) avec de vrais jobs terminés — ce
// script couvre l'écran dans son état d'ENTRÉE (celui atteint depuis la
// navigation), pas la profondeur de chaque état applicatif possible.
const AUTH_TOKEN = process.argv[2];
const THEMES = ["graphite", "ardoise", "minuit", "ivoire", "porcelaine"];

const PUBLIC_SCREENS = [
  { name: "login", url: "/login" },
  { name: "register", url: "/register" },
];
const AUTH_SCREENS = [
  { name: "orientation", url: "/" },
  { name: "onboarding", url: "/onboarding" },
  { name: "dashboard", url: "/dashboard" },
  { name: "profile", url: "/profile" },
  { name: "historique", url: "/historique" },
  { name: "aide", url: "/aide" },
  { name: "datasets", url: "/datasets" },
  { name: "training", url: "/training" },
  { name: "training-history", url: "/training/history" },
  { name: "clustering", url: "/clustering" },
  { name: "reduction-dimension", url: "/reduction-dimension" },
  { name: "anomalies", url: "/anomalies" },
  { name: "non-supervise-historique", url: "/non-supervise/historique" },
  { name: "vision-datasets", url: "/vision/datasets" },
  { name: "vision-classification", url: "/vision/classification" },
  { name: "vision-anomalies", url: "/vision/anomalies" },
  { name: "vision-historique", url: "/vision/historique" },
  { name: "design-system", url: "/design" },
];

const browser = await chromium.launch();
const perScreenTheme = {};
let totalSerious = 0;
const seriousDetails = [];
let totalUndecided = 0;
const undecidedDetails = [];

async function scanPage(page, name, theme) {
  await page.waitForTimeout(700);
  const axeResults = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
  const serious = axeResults.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  // axe classe en `incomplete` — PAS en `violations` — tout contrôle qu'il n'a
  // pas pu trancher seul, notamment color-contrast quand le fond est composé
  // par transparence (nos cartes teintées `bg-accent-N/2`). Ne lire que
  // `violations` faisait donc passer des manquements réels pour un sans-faute :
  // c'est ainsi que le compte Vision de la tuile « Analyses ML » (4,18:1 en
  // porcelaine) n'était pas remonté. On les compte à part — ce ne sont pas des
  // violations avérées — mais on refuse de les ignorer.
  const undecided = axeResults.incomplete.filter((v) => v.id === "color-contrast");
  if (undecided.length > 0) {
    undecidedDetails.push(
      `${name} ${theme}: ${undecided.map((v) => `${v.id} (${v.nodes.length} noeuds à trancher)`).join(", ")}`,
    );
    totalUndecided += undecided.reduce((n, v) => n + v.nodes.length, 0);
  }
  perScreenTheme[`${name}_${theme}`] = serious.length;
  totalSerious += serious.length;
  if (serious.length > 0) {
    seriousDetails.push(`${name} ${theme}: ${serious.map((v) => `${v.id} (${v.nodes.length} noeuds)`).join(", ")}`);
  }
  await page.screenshot({ path: `../_design/captures/lot11-${name}-${theme}.png`, fullPage: false });
}

// --- Écrans publics (pas de jeton) ---
for (const theme of THEMES) {
  const context = await browser.newContext({ viewport: { width: 1512, height: 982 } });
  await context.addInitScript(([t]) => localStorage.setItem("datalab_theme", t), [theme]);
  const page = await context.newPage();
  for (const s of PUBLIC_SCREENS) {
    await page.goto(`http://127.0.0.1:5173${s.url}`, { waitUntil: "networkidle" });
    await scanPage(page, s.name, theme);
  }
  await context.close();
}

// --- Écrans authentifiés ---
for (const theme of THEMES) {
  await fetch("http://127.0.0.1:8000/api/users/me/preferences", {
    method: "PATCH",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${AUTH_TOKEN}` },
    body: JSON.stringify({ ui_theme: theme }),
  });
  const context = await browser.newContext({ viewport: { width: 1512, height: 982 } });
  await context.addInitScript(
    ([token, t]) => {
      localStorage.setItem("datalab_token", token);
      localStorage.setItem("datalab_theme", t);
    },
    [AUTH_TOKEN, theme],
  );
  const page = await context.newPage();
  for (const s of AUTH_SCREENS) {
    await page.goto(`http://127.0.0.1:5173${s.url}`, { waitUntil: "networkidle" });
    await scanPage(page, s.name, theme);
  }
  await context.close();
}

console.log(JSON.stringify({ totalSerious, seriousDetails, totalUndecided, undecidedDetails }, null, 2));
await browser.close();
