import { chromium } from "@playwright/test";

// Vérification clavier Lot 10 (Aide) : la première question de la FAQ est
// focusable et s'ouvre à Enter ; le lien "Aide" de la barre du haut mène
// bien à /aide au clavier.
const AUTH_TOKEN = process.argv[2];

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1512, height: 940 } });
await context.addInitScript(
  ([token]) => {
    localStorage.setItem("datalab_token", token);
    localStorage.setItem("datalab_theme", "graphite");
  },
  [AUTH_TOKEN],
);
const page = await context.newPage();
const results = {};

await page.goto("http://127.0.0.1:5173/dashboard", { waitUntil: "networkidle" });
const helpLink = page.getByRole("link", { name: "Aide" });
await helpLink.focus();
results.helpLinkFocusable = await helpLink.evaluate((el) => el === document.activeElement);
await page.keyboard.press("Enter");
await page.waitForTimeout(500);
results.navigatesToAidePage = page.url().endsWith("/aide");

const firstFaqQuestion = page.getByRole("button", { name: /Mon score est de 0,99/i });
await firstFaqQuestion.focus();
results.faqQuestionFocusable = await firstFaqQuestion.evaluate((el) => el === document.activeElement);
await page.keyboard.press("Enter");
await page.waitForTimeout(300);
results.faqExpandsOnEnter = await page.getByText(/Presque toujours non/i).isVisible().catch(() => false);

console.log(JSON.stringify(results, null, 2));
await browser.close();
