import { chromium } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const AUTH_TOKEN = process.argv[2];
const JOB_ID = process.argv[3] ?? "45";
const THEMES = ["graphite", "ardoise", "minuit", "ivoire", "porcelaine"];

const browser = await chromium.launch();
let totalSerious = 0;

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
  await page.getByRole("tab", { name: /détails/i }).click();
  await page.waitForTimeout(400);

  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
  const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  totalSerious += serious.length;
  console.log(`--- ${theme} --- violations sérieuses/critiques: ${serious.length}`);
  for (const v of serious) {
    console.log(`  [${v.impact}] ${v.id}: ${v.nodes.length} noeud(s)`);
    for (const n of v.nodes.slice(0, 3)) console.log(`    - ${n.html.slice(0, 140)}`);
  }
  await page.close();
}

console.log(`\nTOTAL: ${totalSerious}`);
await browser.close();
