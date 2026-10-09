// Browser check: loads scripts/browser-check.html in headless Chrome (and WebKit if available),
// waits for the engine to init and score the presets, and fails if any request leaves the local server.
//
//   python3 -m http.server 8766 --bind 127.0.0.1    (from the repo root, in another terminal)
//   node scripts/browser-check.mjs [port]
import { chromium, webkit } from 'playwright-core';

const PORT = process.argv[2] ?? '8766';
const ORIGIN = `http://127.0.0.1:${PORT}`;
const SHOT = process.env.SHOT; // optional screenshot path

async function run(name, launch) {
  let browser;
  try { browser = await launch(); } catch (e) { console.log(`${name}: not available (${e.message.split('\n')[0]})`); return null; }
  const context = await browser.newContext({ viewport: { width: 1640, height: 1100 } });
  const page = await context.newPage();
  const requests = [];
  page.on('request', (r) => requests.push(r.url()));
  page.on('pageerror', (e) => console.log(`${name} pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`${name} console.${m.type()}: ${m.text().slice(0, 300)}`); });
  const t0 = Date.now();
  await page.goto(`${ORIGIN}/scripts/browser-check.html`);
  await page.waitForFunction(() => window.__check, null, { timeout: 120000 });
  const check = await page.evaluate(() => window.__check);
  const text = await page.locator('#out').textContent();
  if (SHOT && name === 'chrome') await page.screenshot({ path: SHOT, fullPage: true });
  await browser.close();

  const foreign = requests.filter((u) => !u.startsWith(ORIGIN) && !u.startsWith('blob:') && !u.startsWith('data:'));
  const byPath = [...new Set(requests.map((u) => u.replace(ORIGIN, '')))];
  console.log(`\n=== ${name} (${Date.now() - t0} ms) ===\n${text}`);
  console.log(`requests (${requests.length}):\n  ${byPath.join('\n  ')}`);
  console.log(`off-localhost requests: ${foreign.length ? foreign.join(', ') : 'none'}`);
  return check.ok && foreign.length === 0 && check.same && check.worst > 0.99;
}

const results = {
  chrome: await run('chrome', () => chromium.launch({ channel: 'chrome', headless: true })),
  webkit: await run('webkit', () => webkit.launch({ headless: true })),
};
console.log('\n' + Object.entries(results).map(([k, v]) => `${k}: ${v === null ? 'skipped' : v ? 'PASS' : 'FAIL'}`).join('  '));
process.exit(Object.values(results).some((v) => v === false) || results.chrome !== true ? 1 : 0);
