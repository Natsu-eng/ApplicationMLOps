import { chromium } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

// Lot 10 — captures + axe-core dans les 5 thèmes pour les vues de résultat
// retabulées (Clustering, Anomalies, Réduction de dimension, Vision x2) et
// le Verdict re-repliable (Training), avec de vrais jobs terminés (org 2).
const AUTH_TOKEN = process.argv[2];
const THEMES = ["graphite", "ardoise", "minuit", "ivoire", "porcelaine"];
const PAGES = [
  { name: "training-verdict", url: "/training?job=50", tabName: /Détails/i, isTab: true },
  { name: "clustering", url: "/clustering?job=12" },
  { name: "anomalies", url: "/anomalies?job=7" },
  { name: "dimensionality", url: "/reduction-dimension?job=7" },
  { name: "vision-classification", url: "/vision/classification?job=7" },
  { name: "vision-anomalies", url: "/vision/anomalies?job=5" },
];

const browser = await chromium.launch();
let totalSerious = 0;
const perPage = {};

for (const theme of THEMES) {
  await fetch("http://127.0.0.1:8000/api/users/me/preferences", {
    method: "PATCH",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${AUTH_TOKEN}` },
    body: JSON.stringify({ ui_theme: theme }),
  });
  const context = await browser.newContext({ viewport: { width: 1512, height: 1200 } });
  await context.addInitScript(
    ([token, t]) => {
      localStorage.setItem("datalab_token", token);
      localStorage.setItem("datalab_theme", t);
    },
    [AUTH_TOKEN, theme],
  );
  const page = await context.newPage();

  for (const p of PAGES) {
    await page.goto(`http://127.0.0.1:5173${p.url}`, { waitUntil: "networkidle" });
    await page.waitForTimeout(800);
    if (p.isTab) {
      const tab = page.getByRole("tab", { name: p.tabName });
      if (await tab.isVisible().catch(() => false)) await tab.click();
      await page.waitForTimeout(300);
    }
    const axeResults = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
    const serious = axeResults.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    totalSerious += serious.length;
    perPage[`${p.name}_${theme}`] = serious.length;
    if (serious.length > 0) {
      console.log(`--- axe ${p.name} ${theme} --- ${serious.length} violation(s)`);
      for (const v of serious) console.log(`  [${v.impact}] ${v.id}: ${v.nodes.length} noeud(s) — ${v.nodes[0]?.html.slice(0, 140)}`);
    }
    if (theme === "graphite" || theme === "ivoire") {
      await page.screenshot({ path: `../_design/captures/lot10-${p.name}-${theme}.png`, fullPage: false });
    }
  }
  await page.close();
  await context.close();
}

console.log(JSON.stringify({ perPage, totalSerious }, null, 2));
await browser.close();
