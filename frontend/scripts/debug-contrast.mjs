import { chromium } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const AUTH_TOKEN = process.argv[2];
const THEME = process.argv[3] ?? "ivoire";
const PATH = process.argv[4] ?? "/profile";

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
await fetch("http://127.0.0.1:8000/api/users/me/preferences", {
  method: "PATCH",
  headers: { "Content-Type": "application/json", Authorization: `Bearer ${AUTH_TOKEN}` },
  body: JSON.stringify({ ui_theme: THEME }),
});
await context.addInitScript(
  ([token, theme]) => {
    localStorage.setItem("datalab_token", token);
    localStorage.setItem("datalab_theme", theme);
  },
  [AUTH_TOKEN, THEME]
);
const page = await context.newPage();
await page.goto(`http://127.0.0.1:5173${PATH}`, { waitUntil: "networkidle" });
if (PATH === "/profile") {
  await page.getByText("Préférences", { exact: false }).first().click();
  await page.waitForTimeout(300);
}

const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
const contrast = results.violations.find((v) => v.id === "color-contrast");
console.log(JSON.stringify(contrast, null, 2));

await browser.close();
