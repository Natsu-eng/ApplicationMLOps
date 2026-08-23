import { chromium } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

// Vérification Lot 10 (Aide) — page /aide dans les 5 thèmes, plus
// fonctionnel (FAQ accordéon, filtres catégorie, recherche) et axe-core.
const AUTH_TOKEN = process.argv[2];
const THEMES = ["graphite", "ardoise", "minuit", "ivoire", "porcelaine"];

const browser = await chromium.launch();
const results = {};
let totalSerious = 0;

for (const theme of THEMES) {
  await fetch("http://127.0.0.1:8000/api/users/me/preferences", {
    method: "PATCH",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${AUTH_TOKEN}` },
    body: JSON.stringify({ ui_theme: theme }),
  });
  const context = await browser.newContext({ viewport: { width: 1512, height: 1400 } });
  await context.addInitScript(
    ([token, t]) => {
      localStorage.setItem("datalab_token", token);
      localStorage.setItem("datalab_theme", t);
    },
    [AUTH_TOKEN, theme],
  );
  const page = await context.newPage();
  await page.goto("http://127.0.0.1:5173/aide", { waitUntil: "networkidle" });
  await page.waitForTimeout(600);

  const renderedTheme = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
  if (renderedTheme !== theme) console.error(`THEME MISMATCH: demandé ${theme}, rendu ${renderedTheme}`);

  if (theme === "graphite") {
    results.guideCollapsedInitially = (await page.getByText("Importez vos données", { exact: false }).count()) === 0;
    const guideToggle = page.getByRole("button", { name: "Le parcours, pilier par pilier" });
    await guideToggle.click();
    await page.waitForTimeout(200);
    results.guideStepsVisibleAfterOpen = await page.getByText("Importez vos données", { exact: false }).isVisible().catch(() => false);

    const visionTab = page.getByRole("tab", { name: "Vision" });
    await visionTab.click();
    await page.waitForTimeout(200);
    results.guidePillarTabSwitches = (await page.getByText("Importez vos données", { exact: false }).count()) === 0;

    const firstFaqQuestion = page.getByRole("button", { name: /Mon score est de 0,99/i });
    results.faqCollapsedInitially = (await page.getByText(/Presque toujours non/i).count()) === 0;
    await firstFaqQuestion.click();
    results.faqExpandsOnClick = await page.getByText(/Presque toujours non/i).isVisible().catch(() => false);

    await page.getByRole("button", { name: "Compte" }).click();
    await page.waitForTimeout(200);
    results.categoryFilterWorks = (await page.getByText(/Combien de lignes/i).count()) === 0;

    const search = page.getByLabel("Chercher dans l'aide");
    await search.fill("silhouette");
    await page.waitForTimeout(200);
    results.searchFiltersGlossary = await page.getByText("Score de silhouette", { exact: false }).isVisible().catch(() => false);
    results.searchHidesUnrelatedGlossary = (await page.getByText("Fuite de données", { exact: false }).count()) === 0;
  }

  const axeResults = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
  const serious = axeResults.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  totalSerious += serious.length;
  if (serious.length > 0) {
    console.log(`--- axe ${theme} --- ${serious.length} violation(s)`);
    for (const v of serious) console.log(`  [${v.impact}] ${v.id}: ${v.nodes.length} noeud(s) — ${v.nodes[0]?.html.slice(0, 140)}`);
  }

  await page.screenshot({ path: `../_design/captures/aide-${theme}.png`, fullPage: false });
  await page.close();
}

results.axeTotalSerious = totalSerious;
console.log(JSON.stringify(results, null, 2));
await browser.close();
