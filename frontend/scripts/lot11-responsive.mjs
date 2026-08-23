import { chromium } from "@playwright/test";

// Lot 11 (Vérification finale) — vérification responsive aux 3 largeurs
// demandées (1280/1440/1920), thème graphite, sur les 20 écrans : capture
// + détection de débordement horizontal réel (scrollWidth > clientWidth
// sur <body>, signe d'un élément qui casse la mise en page à cette largeur).
const AUTH_TOKEN = process.argv[2];
const WIDTHS = [1280, 1440, 1920];

const PUBLIC_SCREENS = [
  { name: "login", url: "/login" },
  { name: "register", url: "/register" },
];
const AUTH_SCREENS = [
  { name: "orientation", url: "/" },
  { name: "onboarding", url: "/onboarding" },
  { name: "dashboard", url: "/dashboard" },
  { name: "profile", url: "/profile" },
  { name: "historique", url: "/historique" },
  { name: "aide", url: "/aide" },
  { name: "datasets", url: "/datasets" },
  { name: "training", url: "/training" },
  { name: "training-history", url: "/training/history" },
  { name: "clustering", url: "/clustering" },
  { name: "reduction-dimension", url: "/reduction-dimension" },
  { name: "anomalies", url: "/anomalies" },
  { name: "non-supervise-historique", url: "/non-supervise/historique" },
  { name: "vision-datasets", url: "/vision/datasets" },
  { name: "vision-classification", url: "/vision/classification" },
  { name: "vision-anomalies", url: "/vision/anomalies" },
  { name: "vision-historique", url: "/vision/historique" },
  { name: "design-system", url: "/design" },
];

const browser = await chromium.launch();
const overflowFound = [];

async function checkOverflow(page, name, width) {
  await page.waitForTimeout(500);
  const overflow = await page.evaluate(() => document.body.scrollWidth > document.documentElement.clientWidth + 2);
  if (overflow) overflowFound.push(`${name} @${width}px`);
  await page.screenshot({ path: `../_design/captures/lot11-resp-${name}-${width}.png`, fullPage: false });
}

for (const width of WIDTHS) {
  const context = await browser.newContext({ viewport: { width, height: 900 } });
  const page = await context.newPage();
  for (const s of PUBLIC_SCREENS) {
    await page.goto(`http://127.0.0.1:5173${s.url}`, { waitUntil: "networkidle" });
    await checkOverflow(page, s.name, width);
  }
  await context.close();
}

for (const width of WIDTHS) {
  await fetch("http://127.0.0.1:8000/api/users/me/preferences", {
    method: "PATCH",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${AUTH_TOKEN}` },
    body: JSON.stringify({ ui_theme: "graphite" }),
  });
  const context = await browser.newContext({ viewport: { width, height: 900 } });
  await context.addInitScript(
    ([token]) => {
      localStorage.setItem("datalab_token", token);
      localStorage.setItem("datalab_theme", "graphite");
    },
    [AUTH_TOKEN],
  );
  const page = await context.newPage();
  for (const s of AUTH_SCREENS) {
    await page.goto(`http://127.0.0.1:5173${s.url}`, { waitUntil: "networkidle" });
    await checkOverflow(page, s.name, width);
  }
  await context.close();
}

console.log(JSON.stringify({ overflowFound, totalChecked: WIDTHS.length * (PUBLIC_SCREENS.length + AUTH_SCREENS.length) }, null, 2));
await browser.close();
