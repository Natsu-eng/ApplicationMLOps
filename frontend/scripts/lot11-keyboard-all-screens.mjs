import { chromium } from "@playwright/test";

// Lot 11 (Vérification finale) — passe clavier sur les 20 écrans : Tab
// répété depuis <body>, vérifie qu'aucun focus n'est jamais perdu (piégé
// dans un élément invisible / hors écran) et que la navigation avance
// réellement (l'élément focus change à chaque Tab, pas de double focus
// bloqué). Complète les passes clavier fonctionnelles déjà faites lot par
// lot (Entrée sur un bouton précis, etc.) — ici on vérifie l'HYGIÈNE
// clavier de base sur chaque écran, pas un scénario métier par écran.
const AUTH_TOKEN = process.argv[2];
const TAB_PRESSES = 25;

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
const results = {};

async function checkKeyboardHygiene(page, name) {
  await page.waitForTimeout(500);
  await page.evaluate(() => {
    document.body.focus();
    window.__prevFocusEl = null;
  });
  let stuckCount = 0;
  let invisibleCount = 0;
  for (let i = 0; i < TAB_PRESSES; i++) {
    await page.keyboard.press("Tab");
    const info = await page.evaluate(() => {
      const el = document.activeElement;
      if (!el || el === document.body) return { tag: null, visible: false, same: false };
      const r = el.getBoundingClientRect();
      const visible = r.width > 0 && r.height > 0;
      const same = el === window.__prevFocusEl;
      window.__prevFocusEl = el;
      return { tag: el.tagName, visible, same };
    });
    if (info.same) stuckCount++;
    if (info.tag && !info.visible) invisibleCount++;
  }
  results[name] = { stuckCount, invisibleCount };
}

for (const s of PUBLIC_SCREENS) {
  const context = await browser.newContext({ viewport: { width: 1512, height: 982 } });
  const page = await context.newPage();
  await page.goto(`http://127.0.0.1:5173${s.url}`, { waitUntil: "networkidle" });
  await checkKeyboardHygiene(page, s.name);
  await context.close();
}

const context = await browser.newContext({ viewport: { width: 1512, height: 982 } });
await context.addInitScript(
  ([token]) => {
    localStorage.setItem("datalab_token", token);
    localStorage.setItem("datalab_theme", "graphite");
  },
  [AUTH_TOKEN],
);
const page = await context.newPage();
for (const s of AUTH_SCREENS) {
  await page.goto(`http://127.0.0.1:5173${s.url}`, { waitUntil: "networkidle" });
  await checkKeyboardHygiene(page, s.name);
}
await context.close();

console.log(JSON.stringify(results, null, 2));
await browser.close();
