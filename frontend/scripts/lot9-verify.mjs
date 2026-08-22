import { chromium } from "@playwright/test";

// Vérification Lot 9 (Exploiter et tracer) — carte "Environnement
// d'entraînement" dans l'onglet Détails, dans les 5 thèmes, contre un job
// réel déjà terminé.
const AUTH_TOKEN = process.argv[2];
const JOB_ID = process.argv[3] ?? "45";
const THEMES = ["graphite", "ardoise", "minuit", "ivoire", "porcelaine"];

const browser = await chromium.launch();
const results = {};

for (const theme of THEMES) {
  await fetch("http://127.0.0.1:8000/api/users/me/preferences", {
    method: "PATCH",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${AUTH_TOKEN}` },
    body: JSON.stringify({ ui_theme: theme }),
  });
  const context = await browser.newContext({ viewport: { width: 1512, height: 1300 } });
  await context.addInitScript(
    ([token, t]) => {
      localStorage.setItem("datalab_token", token);
      localStorage.setItem("datalab_theme", t);
    },
    [AUTH_TOKEN, theme],
  );
  const page = await context.newPage();
  await page.goto(`http://127.0.0.1:5173/training?job=${JOB_ID}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(800);

  const renderedTheme = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
  if (renderedTheme !== theme) console.error(`THEME MISMATCH: demandé ${theme}, rendu ${renderedTheme}`);

  await page.getByRole("tab", { name: /détails/i }).click();
  await page.waitForTimeout(400);

  if (theme === "graphite") {
    results.traceabilityCardVisible = await page
      .getByText("Environnement d'entraînement", { exact: false })
      .isVisible()
      .catch(() => false);
    results.seedVisible = await page.getByText("Graine aléatoire", { exact: false }).isVisible().catch(() => false);
    results.sklearnVersionVisible = await page.getByText("sklearn", { exact: false }).isVisible().catch(() => false);
  }

  await page.screenshot({ path: `../_design/captures/details-${theme}.png`, fullPage: false });
  await page.close();
}

console.log(JSON.stringify(results, null, 2));
await browser.close();
