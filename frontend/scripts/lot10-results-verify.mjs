import { chromium } from "@playwright/test";

// Vérification Lot 10 : les vues de résultat retabulées (Clustering, Anomalies,
// Réduction de dimension, Vision classification, Vision anomalies) + le
// ré-repliage du Verdict (Training) + ModelExportActions (artefact + JSON).
const AUTH_TOKEN = process.argv[2];
const BASE = "http://127.0.0.1:5173";

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1512, height: 1000 }, acceptDownloads: true });
await context.addInitScript(
  ([token]) => {
    localStorage.setItem("datalab_token", token);
    localStorage.setItem("datalab_theme", "graphite");
  },
  [AUTH_TOKEN],
);
const page = await context.newPage();
const results = {};

async function checkExportButtons(label) {
  const artifactBtn = page.getByRole("button", { name: "Exporter l'artefact" });
  const configBtn = page.getByRole("button", { name: "Exporter la configuration (JSON)" });
  results[`${label}_exportButtonsVisible`] = (await artifactBtn.isVisible().catch(() => false)) && (await configBtn.isVisible().catch(() => false));
  const [configDownload] = await Promise.all([
    page.waitForEvent("download", { timeout: 5000 }).catch(() => null),
    configBtn.click(),
  ]);
  results[`${label}_configDownloadTriggers`] = configDownload !== null;
  const [artifactDownload] = await Promise.all([
    page.waitForEvent("download", { timeout: 8000 }).catch(() => null),
    artifactBtn.click(),
  ]);
  results[`${label}_artifactDownloadTriggers`] = artifactDownload !== null;
  if (artifactDownload) results[`${label}_artifactFilename`] = artifactDownload.suggestedFilename();
}

// --- Training : job 50, onglet Détails, VerdictClaimItem replié par défaut ---
await page.goto(`${BASE}/training?job=50`, { waitUntil: "networkidle" });
await page.waitForTimeout(600);
const detailsTab = page.getByRole("tab", { name: /Détails/i });
if (await detailsTab.isVisible().catch(() => false)) await detailsTab.click();
await page.waitForTimeout(300);
const firstClaim = page.locator("button[aria-expanded]").first();
results.training_verdictClaimCollapsedInitially = (await firstClaim.getAttribute("aria-expanded")) === "false";
await firstClaim.click();
await page.waitForTimeout(200);
results.training_verdictClaimExpandsOnClick = (await firstClaim.getAttribute("aria-expanded")) === "true";

// --- Clustering : job 12 ---
await page.goto(`${BASE}/clustering?job=12`, { waitUntil: "networkidle" });
await page.waitForTimeout(600);
results.clustering_profilsTabDefault = await page.getByRole("tab", { name: "Profils", selected: true }).count().then((n) => n > 0).catch(() => false);
const comparisonTab = page.getByRole("tab", { name: "Comparaison" });
await comparisonTab.click();
await page.waitForTimeout(300);
results.clustering_comparisonTabSwitches = (await comparisonTab.getAttribute("aria-selected")) === "true";
await checkExportButtons("clustering");

// --- Anomalies : job 7 ---
await page.goto(`${BASE}/anomalies?job=7`, { waitUntil: "networkidle" });
await page.waitForTimeout(600);
results.anomalies_observationsTabDefault = await page.getByRole("tab", { name: "Observations", selected: true }).count().then((n) => n > 0).catch(() => false);
await page.getByRole("tab", { name: "Distribution" }).click();
await page.waitForTimeout(300);
await checkExportButtons("anomalies");

// --- Dimensionality : job 7 (PCA) ---
await page.goto(`${BASE}/reduction-dimension?job=7`, { waitUntil: "networkidle" });
await page.waitForTimeout(600);
results.dimensionality_tabsPresent = await page.getByRole("tab", { name: "Variables contributives" }).isVisible().catch(() => false);
await checkExportButtons("dimensionality");

// --- Vision classification : job 7 ---
await page.goto(`${BASE}/vision/classification?job=7`, { waitUntil: "networkidle" });
await page.waitForTimeout(800);
results.visionClassification_tabsPresent = await page.getByRole("tab", { name: "Grad-CAM" }).isVisible().catch(() => false);
await checkExportButtons("visionClassification");

// --- Vision anomalies : job 5 ---
await page.goto(`${BASE}/vision/anomalies?job=5`, { waitUntil: "networkidle" });
await page.waitForTimeout(800);
results.visionAnomalies_tabsPresent = await page.getByRole("tab", { name: "Exemples" }).isVisible().catch(() => false);
await checkExportButtons("visionAnomalies");

console.log(JSON.stringify(results, null, 2));
await browser.close();
