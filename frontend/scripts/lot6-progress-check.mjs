import { chromium } from "@playwright/test";

// Capture dédiée de l'état "Progression" (Lot 6) dans les 5 thèmes, pendant
// qu'un job réel tourne encore (voir lot6-verify.mjs pour Verdict/Comparaison).
const AUTH_TOKEN = process.argv[2];
const JOB_ID = process.argv[3];
const THEMES = ["graphite", "ardoise", "minuit", "ivoire", "porcelaine"];

const browser = await chromium.launch();
const results = {};

for (const theme of THEMES) {
  // Le serveur gagne sur localStorage (ThemeContext.tsx) et ui_theme a un
  // server_default="graphite" non nul pour TOUT utilisateur — sans ce PATCH,
  // chaque itération se serait silencieusement rendue en graphite malgré la
  // clé localStorage (repli identique à celui documenté dans visual-check.mjs).
  await fetch("http://127.0.0.1:8000/api/users/me/preferences", {
    method: "PATCH",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${AUTH_TOKEN}` },
    body: JSON.stringify({ ui_theme: theme }),
  });
  const context = await browser.newContext({ viewport: { width: 1512, height: 940 } });
  await context.addInitScript(
    ([token, t]) => {
      localStorage.setItem("datalab_token", token);
      localStorage.setItem("datalab_theme", t);
    },
    [AUTH_TOKEN, theme],
  );
  const page = await context.newPage();
  // "networkidle" ne se produit jamais sur cette page tant que le job
  // tourne : le flux SSE (`useJobEvents`) garde une connexion EventSource
  // ouverte en continu — attendre son inactivité réseau bloquerait
  // indéfiniment. "domcontentloaded" + une pause courte suffit.
  await page.goto(`http://127.0.0.1:5173/training?job=${JOB_ID}`, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(2000);

  if (theme === "graphite") {
    results.progressVisible = await page.getByText(/entraînement en cours/i).first().isVisible().catch(() => false);
    results.percentVisible = await page.locator("p.num.text-xs.text-muted-foreground").first().isVisible().catch(() => false);
    results.journalVisible = await page.getByText("Journal de cette session", { exact: false }).isVisible().catch(() => false);
    results.etaVisible = await page.getByText(/reste ≈/i).isVisible().catch(() => false);
  }

  await page.screenshot({ path: `../_design/captures/progression-${theme}.png`, fullPage: true });
  await page.close();
}

console.log(JSON.stringify(results, null, 2));
await browser.close();
