import { chromium } from "@playwright/test";

// Vérifie que le menu "Nouvelle analyse" n'est plus rogné par la carte
// "Vue d'ensemble" (overflow-hidden) après le correctif portail (Lot 10).
const AUTH_TOKEN = process.argv[2];

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1512, height: 900 } });
await context.addInitScript(
  ([token]) => {
    localStorage.setItem("datalab_token", token);
    localStorage.setItem("datalab_theme", "graphite");
  },
  [AUTH_TOKEN],
);
const page = await context.newPage();
await page.goto("http://127.0.0.1:5173/dashboard", { waitUntil: "networkidle" });
await page.waitForTimeout(600);

await page.getByRole("button", { name: /nouvelle analyse/i }).click();
await page.waitForTimeout(300);

const menu = page.getByRole("menu");
const box = await menu.boundingBox();
const supervisedItem = page.getByRole("menuitem", { name: /entraînement/i });

const results = {
  menuVisible: await menu.isVisible().catch(() => false),
  menuItemVisible: await supervisedItem.isVisible().catch(() => false),
  menuFullyOnScreen: box ? box.y >= 0 && box.y + box.height <= 900 && box.x >= 0 && box.x + box.width <= 1512 : false,
};

await page.screenshot({ path: "../_design/captures/dashboard-menu-fix.png", fullPage: false });
console.log(JSON.stringify(results, null, 2));
await browser.close();
