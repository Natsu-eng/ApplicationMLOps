import { chromium } from "@playwright/test";

const AUTH_TOKEN = process.argv[2];

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
await context.addInitScript(
  ([token]) => {
    localStorage.setItem("datalab_token", token);
    localStorage.setItem("datalab_theme", "graphite");
  },
  [AUTH_TOKEN]
);
const page = await context.newPage();
await page.goto("http://127.0.0.1:5173/profile", { waitUntil: "networkidle" });

// 1. Le menu de thème de l'avatar s'ouvre au clic, se ferme à Échap, sans
// piège clavier (le focus revient au bouton qui l'a ouvert).
const themeButton = page.getByRole("button", { name: "Changer de thème" });
await themeButton.click();
await page.waitForTimeout(150);
const menuVisible = await page.getByRole("menu", { name: "Thème d'interface" }).isVisible();
await page.keyboard.press("Escape");
await page.waitForTimeout(150);
const menuClosedAfterEscape = !(await page.getByRole("menu", { name: "Thème d'interface" }).isVisible());

// 2. Onglet "Préférences" atteignable au clavier (Tab depuis le début de
// page) puis activable (Enter), puis les 5 vignettes de thème sont
// atteignables et actionnables au clavier, avec anneau de focus visible.
await page.getByText("Préférences", { exact: false }).first().click();
await page.waitForTimeout(150);

// Navigation clavier RÉELLE (Tab, pas .focus() programmatique — le clic sur
// l'onglet juste avant fausserait l'heuristique :focus-visible de Chromium)
// jusqu'à atteindre la 1re vignette de thème.
let reachedFirstRadio = false;
for (let i = 0; i < 30 && !reachedFirstRadio; i++) {
  await page.keyboard.press("Tab");
  reachedFirstRadio = await page.evaluate(() => document.activeElement?.getAttribute("role") === "radio");
}
const focusOutline = await page.evaluate(() => {
  const el = document.activeElement;
  const s = getComputedStyle(el, null);
  return { outline: s.outlineStyle, outlineWidth: s.outlineWidth, boxShadow: s.boxShadow, tag: el?.tagName, role: el?.getAttribute("role") };
});

await page.keyboard.press("Tab"); // vers la 2e vignette
const secondRadioFocused = await page.evaluate(() => document.activeElement?.getAttribute("role") === "radio");
await page.keyboard.press("Enter");
await page.waitForTimeout(200);
const themeAfterEnter = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));

console.log(
  JSON.stringify(
    {
      menuVisible,
      menuClosedAfterEscape,
      reachedFirstRadio,
      focusOutline,
      secondRadioFocused,
      themeAfterEnter,
    },
    null,
    2
  )
);

await browser.close();
