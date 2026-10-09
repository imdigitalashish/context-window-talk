// Fetches top-voted question titles per "part" from the Stack Exchange API v2.3
// and writes the cleaned corpus to scripts/data/corpus.json.
//
// No API key: the anonymous quota is ~300 requests/day per IP. This script makes
// 2 requests per part, caches every response in scripts/.cache/, and honours the
// API's `backoff` field. Re-running only hits the network for missing pages.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import he from 'he';
import { CANDIDATE_PARTS } from './parts.mjs';

const HERE = new URL('./', import.meta.url);
const CACHE = new URL('.cache/', HERE);
const OUT = new URL('data/corpus.json', HERE);
const PAGES = 2; // 2 × 100 = 200 candidates per part; build-map keeps the top ~130 after cleaning
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let backoffUntil = 0;
let requests = 0;

async function getPage(part, page) {
  const key = `${part.site}-${part.tagged ?? 'all'}-p${page}.json`;
  const cached = new URL(key, CACHE);
  if (existsSync(cached)) return JSON.parse(await readFile(cached, 'utf8'));

  const params = new URLSearchParams({
    page: String(page), pagesize: '100', order: 'desc', sort: 'votes', site: part.site,
  });
  if (part.tagged) params.set('tagged', part.tagged);
  const url = `https://api.stackexchange.com/2.3/questions?${params}`;

  const wait = backoffUntil - Date.now();
  if (wait > 0) { console.log(`  backoff: waiting ${Math.ceil(wait / 1000)}s`); await sleep(wait); }
  await sleep(1200); // stay far below the 30 req/s burst limit

  let res;
  for (let attempt = 1; ; attempt++) {
    try {
      res = await fetch(url, { headers: { 'Accept-Encoding': 'gzip' } });
      break;
    } catch (e) { // transient network errors (ECONNRESET etc.); API errors are handled below
      if (attempt >= 4) throw e;
      console.log(`  network error (${e.cause?.code ?? e.message}), retry ${attempt} in ${attempt * 3}s`);
      await sleep(attempt * 3000);
    }
  }
  requests++;
  const body = await res.json();
  if (!res.ok || body.error_id) {
    throw new Error(`${url} -> HTTP ${res.status} ${body.error_id ?? ''} ${body.error_name ?? ''} ${body.error_message ?? ''}`);
  }
  if (body.backoff) backoffUntil = Date.now() + body.backoff * 1000 + 500;
  console.log(`  GET ${part.site}${part.tagged ? '/' + part.tagged : ''} p${page}: ${body.items.length} items, quota ${body.quota_remaining}/${body.quota_max}${body.backoff ? `, backoff ${body.backoff}s` : ''}`);
  await writeFile(cached, JSON.stringify(body));
  return body;
}

// Titles on these sites are English, but reject anything carrying non-Latin scripts
// or a high share of non-ASCII letters.
const NON_LATIN = /[\u0370-\u03FF\u0400-\u052F\u0590-\u05FF\u0600-\u06FF\u0900-\u0DFF\u0E00-\u0E7F\u1100-\u11FF\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]/;
function isEnglish(title) {
  if (NON_LATIN.test(title)) return false;
  const letters = title.match(/\p{L}/gu) ?? [];
  if (letters.length < 8) return false;
  const ascii = letters.filter((c) => c <= '\u007f').length;
  return ascii / letters.length >= 0.95;
}

const norm = (t) => t.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();

async function main() {
  await mkdir(CACHE, { recursive: true });
  await mkdir(new URL('data/', HERE), { recursive: true });

  const perPart = [];
  for (const part of CANDIDATE_PARTS) {
    const items = [];
    const sources = [{ site: part.site, tagged: part.tagged, pages: PAGES }, ...(part.extra ?? []).map((e) => ({ site: part.site, ...e }))];
    for (const src of sources) {
      for (let page = 1; page <= src.pages; page++) {
        const body = await getPage(src, page);
        items.push(...body.items);
        if (!body.items.length) break; // has_more is unreliable (it is false on full tags), so only stop on empty pages
      }
    }
    perPart.push({ part, items });
  }

  // Clean, then drop titles that appear in more than one part (ambiguous by definition).
  const seenIn = new Map();
  const cleaned = perPart.map(({ part, items }) => {
    const byId = new Map();
    for (const q of items) {
      const title = he.decode(q.title).replace(/\s+/g, ' ').trim();
      if (!isEnglish(title) || title.length > 160) continue;
      if (part.keep && !part.keep.test(title)) continue; // topical filter for noisy tags
      if (!byId.has(q.question_id)) byId.set(q.question_id, { id: q.question_id, score: q.score, title, link: q.link });
    }
    const unique = new Map();
    for (const q of byId.values()) if (!unique.has(norm(q.title))) unique.set(norm(q.title), q);
    for (const k of unique.keys()) seenIn.set(k, (seenIn.get(k) ?? 0) + 1);
    return { part, unique };
  });

  const out = { fetchedAt: new Date().toISOString(), source: 'Stack Exchange API v2.3 (top-voted questions), CC BY-SA 4.0', parts: [] };
  for (const { part, unique } of cleaned) {
    const qs = [...unique.entries()].filter(([k]) => seenIn.get(k) === 1).map(([, q]) => q).sort((a, b) => b.score - a.score);
    const { keep, ...meta } = part;
    out.parts.push({ ...meta, ...(keep ? { keep: String(keep) } : {}), count: qs.length, questions: qs });
    console.log(`${part.id.padEnd(14)} ${String(qs.length).padStart(4)} titles  e.g. "${qs[0]?.title}"`);
  }
  await writeFile(OUT, JSON.stringify(out, null, 1));
  console.log(`\n${requests} API requests this run. Wrote ${OUT.pathname}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
