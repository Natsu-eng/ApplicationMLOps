import { chromium } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const AUTH_TOKEN = process.argv[2];
const RULE = process.argv[3];
const PATH = process.argv[4] ?? "/dev/components";

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
await page.goto(`http://127.0.0.1:5173${PATH}`, { waitUntil: "networkidle" });

const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]).analyze();
const violation = results.violations.find((v) => v.id === RULE);
if (!violation) {
  console.log(`Pas de violation "${RULE}" trouvée.`);
} else {
  console.log(JSON.stringify(violation.nodes.map((n) => ({ html: n.html, target: n.target })), null, 2));
}

await browser.close();
