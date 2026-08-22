import { chromium } from "@playwright/test";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto("http://127.0.0.1:5173/login", { waitUntil: "networkidle" });
await page.evaluate(() => document.documentElement.setAttribute("data-theme", "graphite"));

const result = await page.evaluate(() => {
  function parseRgb(cssColor) {
    const probeEl = document.createElement("span");
    probeEl.style.display = "none";
    document.body.appendChild(probeEl);
    probeEl.style.color = cssColor;
    const resolved = getComputedStyle(probeEl).color;

    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = resolved;
    ctx.fillRect(0, 0, 1, 1);
    const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
    document.body.removeChild(probeEl);
    return { resolved, rgb: [r, g, b] };
  }
  function relativeLuminance([r, g, b]) {
    const toLinear = (c) => {
      const s = c / 255;
      return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    };
    const [rl, gl, bl] = [toLinear(r), toLinear(g), toLinear(b)];
    return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl;
  }

  const out = {};
  for (const pct of [69, 80, 92, 96]) {
    const css = `color-mix(in oklch, var(--info) ${pct}%, var(--surface))`;
    const { resolved, rgb } = parseRgb(css);
    out[pct] = { resolved, rgb, luminance: relativeLuminance(rgb) };
  }
  out.infoRaw = parseRgb("var(--info)");
  out.surfaceRaw = parseRgb("var(--surface)");
  return out;
});

console.log(JSON.stringify(result, null, 2));
await browser.close();
