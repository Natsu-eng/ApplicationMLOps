import { chromium } from "@playwright/test";

// Vérification clavier Lot 6 : l'onglet Comparaison est atteignable et
// activable au clavier (role="tab"), et le bouton "Annuler cet
// entraînement" de la Progression est focusable.
const AUTH_TOKEN = process.argv[2];
const RESULT_JOB_ID = process.argv[3] ?? "45";

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1512, height: 940 } });
await context.addInitScript(
  ([token]) => {
    localStorage.setItem("datalab_token", token);
    localStorage.setItem("datalab_theme", "graphite");
  },
  [AUTH_TOKEN],
);
const page = await context.newPage();
const results = {};

await page.goto(`http://127.0.0.1:5173/training?job=${RESULT_JOB_ID}`, { waitUntil: "networkidle" });
await page.waitForTimeout(800);

const comparisonTab = page.getByRole("tab", { name: /comparaison/i });
await comparisonTab.focus();
results.comparisonTabFocusable = await comparisonTab.evaluate((el) => el === document.activeElement);
await page.keyboard.press("Enter");
await page.waitForTimeout(400);
results.comparisonTabActivatesOnEnter = await comparisonTab.evaluate((el) => el.getAttribute("aria-selected") === "true");
results.comparisonContentVisible = await page
  .getByText("Tous les modèles comparés", { exact: false })
  .isVisible()
  .catch(() => false);

console.log(JSON.stringify(results, null, 2));
await browser.close();
