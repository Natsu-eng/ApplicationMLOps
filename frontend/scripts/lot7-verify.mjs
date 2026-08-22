import { chromium } from "@playwright/test";

// Vérification Lot 7 (Non supervisé) — panneau "Où placer le curseur"
// (Anomalies) dans les 5 thèmes, contre un job réel déjà terminé.
const AUTH_TOKEN = process.argv[2];
const JOB_ID = process.argv[3] ?? "8";
const THEMES = ["graphite", "ardoise", "minuit", "ivoire", "porcelaine"];

const browser = await chromium.launch();
const results = {};

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
  await page.goto(`http://127.0.0.1:5173/anomalies?job=${JOB_ID}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(800);

  const renderedTheme = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
  if (renderedTheme !== theme) console.error(`THEME MISMATCH: demandé ${theme}, rendu ${renderedTheme}`);

  if (theme === "graphite") {
    const slider = page.getByRole("slider", { name: /seuil exploratoire/i });
    results.sliderVisible = await slider.isVisible().catch(() => false);
    await slider.focus();
    results.sliderFocusable = await slider.evaluate((el) => el === document.activeElement).catch(() => false);
    const before = await page.getByText(/ligne.*au-dessus/i).textContent().catch(() => "");
    await slider.press("End");
    await page.waitForTimeout(150);
    const after = await page.getByText(/ligne.*au-dessus/i).textContent().catch(() => "");
    results.sliderChangesCount = before !== after;
  }

  await page.screenshot({ path: `../_design/captures/anomalies-${theme}.png`, fullPage: false });
  await page.close();
}

console.log(JSON.stringify(results, null, 2));
await browser.close();
