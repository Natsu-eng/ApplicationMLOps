#!/usr/bin/env node
/**
 * Capture d'écran + audit axe-core d'un ensemble d'écrans, dans les 5 thèmes
 * (SPEC-UI.md §2) — script réutilisé à chaque lot de la refonte visuelle,
 * pas seulement le Lot 1 (voir _design/JOURNAL.md).
 *
 * Usage :
 *   node scripts/visual-check.mjs <config.json>
 *
 * Le fichier de config est une liste d'écrans :
 *   [{ "name": "Login", "path": "/login", "auth": false, "beforeShot": null }, ...]
 *
 * Variables d'environnement :
 *   BASE_URL   (défaut http://127.0.0.1:5173)
 *   AUTH_TOKEN (token JWT pour les écrans authentifiés — voir "auth": true)
 *   OUT_DIR    (défaut ../../_design/captures)
 */
import { chromium } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BASE_URL = process.env.BASE_URL ?? "http://127.0.0.1:5173";
const API_BASE = process.env.API_BASE ?? "http://127.0.0.1:8000";
const AUTH_TOKEN = process.env.AUTH_TOKEN ?? "";
const OUT_DIR = process.env.OUT_DIR ?? path.resolve(__dirname, "../../_design/captures");
const RESULTS_PATH = process.env.RESULTS_PATH ?? path.join(OUT_DIR, "_results.json");
const THEMES = ["graphite", "ivoire", "minuit", "ardoise", "porcelaine"];

const configPath = process.argv[2];
if (!configPath) {
  console.error("Usage: node scripts/visual-check.mjs <config.json>");
  process.exit(1);
}
const screens = JSON.parse(readFileSync(configPath, "utf-8"));

mkdirSync(OUT_DIR, { recursive: true });

function slug(s) {
  return s
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .toLowerCase();
}

async function main() {
  const browser = await chromium.launch();
  const results = [];

  for (const screen of screens) {
    for (const theme of THEMES) {
      const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
      if (screen.auth) {
        // Le serveur gagne sur localStorage (ordre de résolution voulu,
        // ThemeContext.tsx) : sans ceci, la préférence déjà enregistrée sur
        // le compte de test écraserait le thème qu'on veut justement capturer.
        await fetch(`${API_BASE}/api/users/me/preferences`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${AUTH_TOKEN}` },
          body: JSON.stringify({ ui_theme: theme }),
        });
        await context.addInitScript(
          ([token, themeName]) => {
            localStorage.setItem("datalab_token", token);
            localStorage.setItem("datalab_theme", themeName);
          },
          [AUTH_TOKEN, theme]
        );
      } else {
        await context.addInitScript((themeName) => {
          localStorage.setItem("datalab_theme", themeName);
        }, theme);
      }
      const page = await context.newPage();
      await page.goto(`${BASE_URL}${screen.path}`, { waitUntil: "networkidle" });
      // Confirme que le thème demandé est bien celui rendu (pas de repli
      // silencieux vers graphite si la clé localStorage n'a pas pris).
      const renderedTheme = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
      for (const text of screen.clickText ?? []) {
        // Cherche d'abord par nom accessible (aria-label, ex. bouton icône
        // seul), puis par texte visible (ex. onglet) — couvre les deux cas
        // sans configuration supplémentaire par écran.
        const byRole = page.getByRole("button", { name: text });
        const locator = (await byRole.count()) > 0 ? byRole.first() : page.getByText(text, { exact: false }).first();
        await locator.click();
        await page.waitForTimeout(200);
      }
      const fileName = `${slug(screen.name)}-${theme}.png`;
      await page.screenshot({ path: path.join(OUT_DIR, fileName), fullPage: false });

      const axeResults = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze();
      const serious = axeResults.violations.filter((v) => v.impact === "serious" || v.impact === "critical");

      results.push({
        screen: screen.name,
        theme,
        renderedTheme,
        file: fileName,
        axeViolationsSerious: serious.map((v) => ({ id: v.id, impact: v.impact, nodes: v.nodes.length })),
        axeViolationsOther: axeResults.violations.length - serious.length,
      });

      await context.close();
    }
  }

  await browser.close();
  writeFileSync(RESULTS_PATH, JSON.stringify(results, null, 2));
  console.log(`Résultats écrits dans ${RESULTS_PATH}`);

  const failed = results.filter((r) => r.axeViolationsSerious.length > 0 || r.renderedTheme !== r.theme);
  console.log(`${results.length} capture(s), ${failed.length} en échec.`);
  if (failed.length > 0) {
    console.log(JSON.stringify(failed, null, 2));
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
