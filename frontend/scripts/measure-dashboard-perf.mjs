import { chromium } from "@playwright/test";

const AUTH_TOKEN = process.argv[2];

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1512, height: 940 } });
await context.addInitScript(
  ([token]) => {
    localStorage.setItem("datalab_token", token);
    localStorage.setItem("datalab_theme", "graphite");
  },
  [AUTH_TOKEN]
);
const page = await context.newPage();

const apiTimings = [];
page.on("response", (res) => {
  if (res.url().includes("/api/")) {
    apiTimings.push({ url: res.url().replace(/^.*\/api/, "/api"), status: res.status() });
  }
});

const start = Date.now();
await page.goto("http://127.0.0.1:5173/dashboard", { waitUntil: "domcontentloaded" });
const domContentLoaded = Date.now() - start;

// Contenu réel affiché = le titre "Bonjour" visible (pas juste le squelette).
await page.getByText("Bonjour,", { exact: false }).waitFor({ state: "visible", timeout: 15000 });
const realContentVisible = Date.now() - start;

// Fin de tout chargement réseau.
await page.waitForLoadState("networkidle");
const networkIdle = Date.now() - start;

console.log(
  JSON.stringify(
    {
      domContentLoadedMs: domContentLoaded,
      realContentVisibleMs: realContentVisible,
      networkIdleMs: networkIdle,
      apiCalls: apiTimings,
    },
    null,
    2
  )
);

await browser.close();
