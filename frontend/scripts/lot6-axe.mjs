import { chromium } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

// Scan axe-core complet (Lot 6) sur les onglets Performance et Comparaison
// de la vue Résultats, dans les 5 thèmes — les deux zones touchées par ce
// lot. Le thème doit être posé via la préférence SERVEUR (PATCH), pas
// seulement localStorage : ui_theme a un server_default="graphite" non nul
// pour tout utilisateur, et le serveur gagne sur localStorage
// (ThemeContext.tsx) — sans ce PATCH, toutes les itérations se
// rendraient dans le dernier thème persisté côté serveur.
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
  const context = await browser.newContext({ viewport: { width: 1512, height: 1200 } });
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

  const renderedTheme = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
  if (renderedTheme !== theme) console.error(`THEME MISMATCH: demandé ${theme}, rendu ${renderedTheme}`);

  const perfResults = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();

  await page.getByRole("tab", { name: /comparaison/i }).click();
  await page.waitForTimeout(500);
  const compResults = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();

  function summarize(label, results) {
    const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    totalSerious += serious.length;
    console.log(`--- ${theme} / ${label} --- violations sérieuses/critiques: ${serious.length}`);
    for (const v of serious) {
      console.log(`  [${v.impact}] ${v.id}: ${v.nodes.length} noeud(s)`);
      for (const n of v.nodes.slice(0, 3)) {
        console.log(`    - ${n.html.slice(0, 140)}`);
        if (n.any?.[0]?.message) console.log(`      ${n.any[0].message}`);
      }
    }
  }
  summarize("Performance", perfResults);
  summarize("Comparaison", compResults);
  await page.close();
}

console.log(`\nTOTAL violations sérieuses/critiques sur les 5 thèmes: ${totalSerious}`);
await browser.close();
