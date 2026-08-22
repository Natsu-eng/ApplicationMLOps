import { chromium } from "@playwright/test";

const AUTH_TOKEN = process.argv[2];
const PATH = process.argv[3] ?? "/dashboard";
const THEME = process.argv[4] ?? "graphite";
const OUT = process.argv[5] ?? "screenshot.png";

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1512, height: 940 } });
await context.addInitScript(
  ([token, theme]) => {
    localStorage.setItem("datalab_token", token);
    localStorage.setItem("datalab_theme", theme);
  },
  [AUTH_TOKEN, THEME]
);
const page = await context.newPage();
await page.goto(`http://127.0.0.1:5173${PATH}`, { waitUntil: "networkidle" });
await page.waitForTimeout(300);
await page.screenshot({ path: OUT, fullPage: false });
await browser.close();
console.log(`Saved ${OUT}`);
