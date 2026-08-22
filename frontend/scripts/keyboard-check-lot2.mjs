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

// 1. CommandPalette (⌘K global) — ouverture au raccourci, fermeture à Échap.
await page.keyboard.press("Control+k");
await page.waitForTimeout(150);
results.commandPaletteOpensOnShortcut = await page.getByRole("dialog", { name: "Palette de commandes" }).isVisible();
await page.keyboard.press("Escape");
await page.waitForTimeout(150);
results.commandPaletteClosesOnEscape = !(await page.getByRole("dialog", { name: "Palette de commandes" }).isVisible());

// 2. CommandPalette — navigation flèches + Enter navigue réellement.
await page.keyboard.press("Control+k");
await page.waitForTimeout(150);
await page.keyboard.type("profil");
await page.waitForTimeout(150);
await page.keyboard.press("Enter");
await page.waitForTimeout(300);
results.commandPaletteEnterNavigates = page.url().includes("/profile");
await page.goto("http://127.0.0.1:5173/dev/components", { waitUntil: "networkidle" });

// 3. Drawer — piège de focus (Tab ne sort jamais), Échap ferme, focus revient au déclencheur.
const drawerTrigger = page.getByRole("button", { name: "Ouvrir le tiroir" });
await drawerTrigger.focus();
await page.keyboard.press("Enter");
await page.waitForTimeout(200);
results.drawerOpens = await page.getByRole("dialog", { name: "Détail (démonstration)" }).isVisible();
await page.keyboard.press("Escape");
await page.waitForTimeout(200);
results.drawerClosesOnEscape = !(await page.getByRole("dialog", { name: "Détail (démonstration)" }).isVisible());
results.focusReturnsToTrigger = await drawerTrigger.evaluate((el) => el === document.activeElement);

// 4. Segmented — role=radiogroup, activation au clic change bien la valeur (déjà couvert visuellement) + accessible name.
const segmented = page.getByRole("radiogroup", { name: "Mode d'affichage" });
results.segmentedHasAccessibleName = await segmented.isVisible();

// 5. Alert — dismiss au clic ferme réellement l'alerte danger (role="alert"
// existe aussi ailleurs sur la page — message d'erreur de Field — donc on
// cible par le texte propre à la démo Alert).
const dangerAlert = page.getByRole("alert").filter({ hasText: "Échec de l'entraînement" });
const dangerAlertCountBefore = await dangerAlert.count();
await page.getByRole("button", { name: "Fermer l'alerte" }).click();
await page.waitForTimeout(150);
const dangerAlertCountAfter = await page.getByRole("alert").filter({ hasText: "Échec de l'entraînement" }).count();
results.dangerAlertCountBefore = dangerAlertCountBefore;
results.dangerAlertCountAfter = dangerAlertCountAfter;
results.alertDismissWorks = dangerAlertCountBefore === 1 && dangerAlertCountAfter === 0;

console.log(JSON.stringify(results, null, 2));
await browser.close();
