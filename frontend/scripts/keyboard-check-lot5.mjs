import { chromium } from "@playwright/test";

const AUTH_TOKEN = process.argv[2];
const DATASET_NAME = process.argv[3] ?? "test_quality.csv";

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1512, height: 940 } });
await context.addInitScript(
  ([token]) => {
    localStorage.setItem("datalab_token", token);
    localStorage.setItem("datalab_theme", "graphite");
  },
  [AUTH_TOKEN]
);
const page = await context.newPage();
const results = {};

await page.goto("http://127.0.0.1:5173/datasets", { waitUntil: "networkidle" });

// Ouvre le modal EDA du dataset de test (bouton "Explorer" sur sa carte) —
// organisation de test fraîche, un seul dataset : le premier bouton
// "Explorer" de la page suffit, pas besoin de remonter jusqu'à la carte.
await page.getByText(DATASET_NAME, { exact: false }).first().waitFor({ state: "visible", timeout: 10000 });
await page.getByRole("button", { name: /explorer/i }).first().click();
await page.waitForTimeout(400);

// Onglet Qualité des données (composant Tabs -> role="tab", pas "button").
await page.getByRole("tab", { name: /Qualité des données/i }).click();
await page.waitForTimeout(1500); // le contrôle qualité fait un vrai appel réseau

results.qualityTabOpened = await page.getByText("La question à se poser", { exact: false }).first().isVisible().catch(() => false);

const keepButton = page.getByRole("button", { name: "Garder tel quel" }).first();
const keepVisible = await keepButton.isVisible().catch(() => false);
results.keepButtonVisible = keepVisible;
if (keepVisible) {
  await keepButton.focus();
  results.keepButtonFocusable = await keepButton.evaluate((el) => el === document.activeElement);
  await page.keyboard.press("Enter");
  await page.waitForTimeout(200);
  results.keepButtonActivatesOnEnter = (await page.getByText("Conservée telle quelle").count()) > 0;
}

const excludeButton = page.getByRole("button", { name: /Exclure «/ }).first();
results.excludeButtonVisible = await excludeButton.isVisible().catch(() => false);

console.log(JSON.stringify(results, null, 2));
await browser.close();
