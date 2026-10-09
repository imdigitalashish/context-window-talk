// Headless Chrome check of the page: console errors, off-localhost requests, and screenshots.
//   cd u-shape/.cache/node && npm i playwright-core@1     (once; uses the installed Google Chrome)
//   node u-shape/experiments/check-page.mjs [url]
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(here, '../.cache/node/package.json'));
const { chromium } = require('playwright-core');

const url = process.argv[2] ?? 'http://localhost:8000/u-shape/';
const screens = path.join(here, '../screens');
const shots = (process.env.SHOTS ?? 'start,middle,end,fix,2k').split(',');

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.emulateMedia({ reducedMotion: 'reduce' }); // no transitions -> stable screenshots
const problems = [], external = [], requests = [];
page.on('console', m => { if (['error', 'warning'].includes(m.type())) problems.push(`${m.type()}: ${m.text()}`); });
page.on('pageerror', e => problems.push(`pageerror: ${e.message}`));
page.on('request', r => {
  const u = new URL(r.url());
  requests.push(r.url());
  if (!['localhost', '127.0.0.1'].includes(u.hostname) && u.protocol !== 'data:') external.push(r.url());
});
page.on('requestfailed', r => problems.push(`requestfailed: ${r.url()} ${r.failure()?.errorText}`));
page.on('response', r => { if (r.status() >= 400) problems.push(`HTTP ${r.status()}: ${r.url()}`); });

await page.goto(url, { waitUntil: 'networkidle' });
await page.waitForSelector('#stage[data-ready=true]', { timeout: 10000 });
const press = async (k, n = 1) => { for (let i = 0; i < n; i++) await page.keyboard.press(k); };
const shot = async name => {
  await page.waitForTimeout(150);
  await page.screenshot({ path: path.join(screens, `${name}.png`) });
  const r = await page.evaluate(() => ({
    where: document.getElementById('where').textContent,
    hits: document.getElementById('hits').textContent + document.getElementById('outOf').textContent,
    note: document.getElementById('missNote').textContent,
    sink: document.getElementById('sinkPct').textContent + '%',
    len: document.querySelector('#lenSeg [aria-pressed=true]')?.textContent,
    look: document.getElementById('lookBtn').getAttribute('aria-pressed'),
    fix: document.getElementById('fixBtn').getAttribute('aria-pressed'),
  }));
  console.log(name.padEnd(16), JSON.stringify(r));
};

await press('Escape');
if (shots.includes('start')) { await press('Home'); await shot('needle-start'); }
if (shots.includes('middle')) { await press('Home'); await press('ArrowRight', 5); await shot('needle-middle'); }
if (shots.includes('end')) { await press('End'); await shot('needle-end'); }
if (shots.includes('fix')) {
  await press('Home'); await press('ArrowRight', 5); await press('q');
  await shot('fix-middle');
  await press('q');
}
if (shots.includes('single')) { await press('Home'); await press('ArrowRight', 5); await press('l'); await shot('single-middle'); await press('l'); }
if (shots.includes('2k')) { await press('1'); await press('Home'); await press('ArrowRight', 5); await shot('2k-middle'); await press('Escape'); }

// overflow check: any text box poking outside the stage or its card
const overflow = await page.evaluate(() => {
  const out = [];
  for (const e of document.querySelectorAll('#stage *')) {
    if (!(e instanceof HTMLElement) || !e.offsetParent) continue;
    if (e.scrollWidth > e.clientWidth + 1 && getComputedStyle(e).overflow === 'visible' && e.childElementCount === 0 && e.clientWidth > 0)
      out.push(`${e.tagName.toLowerCase()}#${e.id || e.className}: ${e.scrollWidth}>${e.clientWidth}`);
  }
  return out;
});
console.log(JSON.stringify({ problems, external, requests: requests.length, overflow }, null, 1));
await browser.close();
process.exit(problems.length || external.length ? 1 : 0);
