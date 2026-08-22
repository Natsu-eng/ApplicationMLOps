import { chromium } from "@playwright/test";

const AUTH_TOKEN = process.argv[2];

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
await context.addInitScript(
  ([token]) => {
    localStorage.setItem("datalab_token", token);
    localStorage.setItem("datalab_theme", "graphite");
  },
  [AUTH_TOKEN]
);
const page = await context.newPage();
await page.goto("http://127.0.0.1:5173/dev/components", { waitUntil: "networkidle" });

const results = {};

// 1. RocPr : la bascule ROC/PR change réellement le titre affiché.
const titleBefore = await page.getByText("Courbe ROC", { exact: false }).first().isVisible();
await page.getByRole("radio", { name: "Précision-rappel" }).click();
await page.waitForTimeout(150);
results.rocPrTitleBefore = titleBefore;
results.rocPrSwitchesToPR = await page.getByText("Courbe précision-rappel", { exact: false }).first().isVisible();

// 2. ChartFrame : le tableau de repli s'ouvre/se ferme au clic ET reste accessible au clavier (Tab + Enter).
const toggleButtons = page.getByRole("button", { name: "Voir les données en tableau" });
const firstToggle = toggleButtons.first();
await firstToggle.focus();
const toggleFocused = await firstToggle.evaluate((el) => el === document.activeElement);
await page.keyboard.press("Enter");
await page.waitForTimeout(150);
results.tableToggleFocusable = toggleFocused;
results.tableToggleOpensOnEnter = (await page.getByRole("button", { name: "Masquer le tableau de données" }).count()) > 0;

console.log(JSON.stringify(results, null, 2));
await browser.close();
