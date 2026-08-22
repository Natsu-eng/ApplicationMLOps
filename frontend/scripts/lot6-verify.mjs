import { chromium } from "@playwright/test";

// Vérification Lot 6 (Supervisé) — Verdict + Comparaison d'un job réel déjà
// terminé, dans les 5 thèmes. Capture des captures d'écran réelles plus une
// vérification fonctionnelle du contenu (question/evidence line, onglet
// Comparaison).
const AUTH_TOKEN = process.argv[2];
const JOB_ID = process.argv[3] ?? "45";
const THEMES = ["graphite", "ardoise", "minuit", "ivoire", "porcelaine"];

const browser = await chromium.launch();
const results = {};

for (const theme of THEMES) {
  // Le serveur gagne sur localStorage (ThemeContext.tsx) et ui_theme a un
  // server_default="graphite" non nul pour TOUT utilisateur — sans ce PATCH,
  // chaque itération se serait silencieusement rendue en graphite malgré la
  // clé localStorage (même repli que documenté dans visual-check.mjs).
  await fetch("http://127.0.0.1:8000/api/users/me/preferences", {
    method: "PATCH",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${AUTH_TOKEN}` },
    body: JSON.stringify({ ui_theme: theme }),
  });

  // Viewport haut plutôt que fullPage (comme visual-check.mjs) : une capture
  // fullPage sur une page qui dépasse la hauteur de l'écran duplique la
  // barre latérale/topbar `position:fixed` d'AppShell à chaque "segment"
  // recollé par Playwright — artefact de l'outil de capture, jamais un bug
  // de rendu réel (au défilement normal dans un navigateur, la barre suit
  // simplement le viewport).
  const context = await browser.newContext({ viewport: { width: 1512, height: 1400 } });
  await context.addInitScript(
    ([token, t]) => {
      localStorage.setItem("datalab_token", token);
      localStorage.setItem("datalab_theme", t);
    },
    [AUTH_TOKEN, theme],
  );
  const page = await context.newPage();
  await page.goto(`http://127.0.0.1:5173/training?job=${JOB_ID}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(1000);

  const renderedTheme = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
  if (renderedTheme !== theme) {
    console.error(`THEME MISMATCH: demandé ${theme}, rendu ${renderedTheme}`);
  }

  await page.screenshot({ path: `../_design/captures/verdict-${theme}.png`, fullPage: false });

  if (theme === "graphite") {
    results.verdictVisible = await page.getByText("Verdict", { exact: true }).first().isVisible().catch(() => false);
    results.evidenceLineVisible = await page.locator("p.num.text-caption").first().isVisible().catch(() => false);
  }

  await page.getByRole("tab", { name: /comparaison/i }).click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `../_design/captures/comparaison-${theme}.png`, fullPage: false });

  if (theme === "graphite") {
    results.comparisonTabVisible = await page
      .getByText("Tous les modèles comparés", { exact: false })
      .isVisible()
      .catch(() => false);
    results.comparisonRowCount = await page.locator("table tbody tr").count().catch(() => 0);
  }
  await page.close();
}

console.log(JSON.stringify(results, null, 2));
await browser.close();
