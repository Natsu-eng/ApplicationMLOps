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
const results = {};

// 1. Menu de theme de l'avatar (dashboard, rail flottant).
await page.goto("http://127.0.0.1:5173/dashboard", { waitUntil: "networkidle" });
const themeButton = page.getByRole("button", { name: "Changer de thème" });
await themeButton.focus();
await page.keyboard.press("Enter");
await page.waitForTimeout(150);
results.themeMenuOpensFromRail = await page.getByRole("menu", { name: "Thème d'interface" }).isVisible();
await page.keyboard.press("Escape");
await page.waitForTimeout(150);
results.themeMenuClosesOnEscape = !(await page.getByRole("menu", { name: "Thème d'interface" }).isVisible());

// 2. Onboarding : la dropzone (label) est atteignable au clavier (input file focusable via label).
await page.goto("http://127.0.0.1:5173/onboarding", { waitUntil: "networkidle" });
const fileInput = page.locator('input[type="file"]');
await fileInput.focus();
results.dropzoneInputFocusable = await fileInput.evaluate((el) => el === document.activeElement);
results.skipLinkPresent = await page.getByRole("button", { name: "Passer cette étape" }).isVisible();

console.log(JSON.stringify(results, null, 2));
await browser.close();
